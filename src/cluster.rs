//! Stage a compiled model onto a heterogeneous cluster.
//!
//! Each device has a kind, a buffer budget, and a code budget. Parameter
//! bytes are homed in buffer memory. The live tape is homed there too. A
//! compute window runs on one device only when its resident bytes fit the
//! buffer budget that is still free and each kernel phase fits the code
//! budget. Windows tile the batch, the contraction, and the output until
//! that is true. The adjoint of a tile is applied while the tile is
//! resident, so the full adjoint buffer does not have to sit next to the
//! parameters. [`crate::dist::run`] admits one member per device and runs the
//! windows, with the hops carried as messages between members.

use std::collections::VecDeque;

use crate::kernel::{Program, Stage};
use crate::mixtral::{layout, MixtralConfig};

/// Bytes of one CUDA kernel in the code-memory model. A phase may load
/// several kernels when their sum fits the device.
pub const KERNEL_CODE_BYTES: u64 = 8 * 1024;

/// Code budget of the CPU interpreter. A CPU device executes a window when
/// its code budget covers this runtime and its buffer covers the window.
pub const CPU_RUNTIME_BYTES: u64 = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub name: String,
    pub kind: DeviceKind,
    pub buffer_bytes: u64,
    pub code_bytes: u64,
}

/// An undirected link. Bytes move in either direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub a: usize,
    pub b: usize,
}

#[derive(Clone, Debug)]
pub struct Cluster {
    pub devices: Vec<Device>,
    /// Who can exchange a buffer with whom. `Cluster::devices` fills this
    /// from the device list: CPUs form a fabric, each GPU attaches to one
    /// CPU, and GPUs on the same CPU can also reach each other. A cluster
    /// with no CPU peers every GPU.
    pub links: Vec<Link>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    CpuRuntime,
    GemmFwd,
    GemmDx,
    GemmDw,
    GemmDb,
    ReluBwd,
    Add,
    AttnFwd,
    AttnBwd,
    RmsNorm,
    RmsNormBwd,
    Linear,
    LinearDx,
    LinearDw,
    Rope,
    GqaFwd,
    GqaBwd,
    Router,
    RouterBwd,
    Silu,
    SiluBwd,
    Embed,
    EmbedBwd,
    Ce,
    Sgd,
}

impl Kernel {
    pub fn code_bytes(self) -> u64 {
        match self {
            Kernel::CpuRuntime => CPU_RUNTIME_BYTES,
            _ => KERNEL_CODE_BYTES,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kernel::CpuRuntime => "cpu_runtime",
            Kernel::GemmFwd => "fwd_gemm",
            Kernel::GemmDx => "bwd_dx",
            Kernel::GemmDw => "bwd_dw",
            Kernel::GemmDb => "bwd_db",
            Kernel::ReluBwd => "bwd_relu",
            Kernel::Add => "add",
            Kernel::AttnFwd => "fwd_attn",
            Kernel::AttnBwd => "bwd_attn",
            Kernel::RmsNorm => "k_rmsnorm",
            Kernel::RmsNormBwd => "k_rmsnorm_bwd",
            Kernel::Linear => "k_linear",
            Kernel::LinearDx => "k_linear_dx",
            Kernel::LinearDw => "k_linear_dw",
            Kernel::Rope => "k_rope",
            Kernel::GqaFwd => "k_gqa_fwd",
            Kernel::GqaBwd => "k_gqa_bwd",
            Kernel::Router => "k_router",
            Kernel::RouterBwd => "k_router_bwd",
            Kernel::Silu => "k_silu_mul",
            Kernel::SiluBwd => "k_silu_bwd",
            Kernel::Embed => "k_embed",
            Kernel::EmbedBwd => "k_embed_bwd",
            Kernel::Ce => "k_ce",
            Kernel::Sgd => "k_sgd",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomeKind {
    Param,
    Tape,
}

/// A contiguous rectangle of a matrix, or a row-group of a tape, resident
/// on one device.
#[derive(Clone, Debug)]
pub struct Home {
    pub kind: HomeKind,
    pub name: String,
    pub device: usize,
    pub row_begin: u64,
    pub row_end: u64,
    pub col_begin: u64,
    pub col_end: u64,
    pub stride: u64,
}

impl Home {
    pub fn elems(&self) -> u64 {
        self.row_end
            .saturating_sub(self.row_begin)
            .saturating_mul(self.col_end.saturating_sub(self.col_begin))
    }

    pub fn bytes(&self) -> u64 {
        match self.kind {
            HomeKind::Param => f32_bytes(self.elems()),
            HomeKind::Tape => self.elems(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Transfer {
    pub from: usize,
    pub to: usize,
    pub bytes: u64,
}

/// What a hop carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HopKind {
    Param = 0,
    Adjoint = 1,
    Activation = 2,
}

/// One hop of a routed message. A longer route is a sequence of these.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hop {
    pub from: usize,
    pub to: usize,
    pub bytes: u64,
    pub kind: HopKind,
    pub offset: u64,
    pub elems: u64,
    pub row_begin: u64,
    pub row_end: u64,
    pub col_begin: u64,
    pub col_end: u64,
    pub stride: u64,
}

/// Which way a window runs. The schedule is every forward window, then the
/// backward windows in reverse, then the parameter update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Fwd,
    Bwd,
}

/// The operation a window performs. Tile bounds select the rows, columns,
/// and batch items of that operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Exec {
    Embed = 1,
    RmsNorm = 2,
    Linear = 3,
    RopeQ = 4,
    RopeK = 5,
    Gqa = 6,
    AddAttn = 7,
    AddMoe = 8,
    Router = 9,
    Silu = 10,
    Mix = 11,
    Loss = 12,
    Sgd = 13,
}

/// One tiled piece of work. [`crate::dist::run`] runs these in order on `device`.
#[derive(Clone, Debug)]
pub struct Window {
    pub name: String,
    pub device: usize,
    pub code_phases: Vec<Vec<Kernel>>,
    /// Bytes resident on `device` for this window, including homes already
    /// stored there and the transient tape, activations, and adjoint tile.
    pub resident_bytes: u64,
    pub tape_bytes: u64,
    pub param_bytes: u64,
    pub pass: Pass,
    pub exec: Exec,
    pub param_offset: u64,
    pub batch_begin: u64,
    pub batch_end: u64,
    pub row_begin: u64,
    pub row_end: u64,
    pub col_begin: u64,
    pub col_end: u64,
    pub din: u64,
    pub dout: u64,
    pub home: usize,
    pub layer: u32,
    pub expert: u32,
    /// Messages that deliver this window's inputs, in hop order.
    pub inbound: Vec<Hop>,
    /// Messages that carry its outputs back, in hop order.
    pub outbound: Vec<Hop>,
}

#[derive(Clone, Debug)]
pub struct StagePlan {
    pub homes: Vec<Home>,
    pub windows: Vec<Window>,
    pub transfers: Vec<Transfer>,
    pub peak_bytes: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct Compiled {
    pub plan: StagePlan,
    pub gpu_kernels: Vec<Kernel>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageError {
    NoDevice,
    ParamsExceedCluster {
        params: u64,
        capacity: u64,
    },
    TapeExceedsCluster {
        tape: u64,
        free: u64,
    },
    TileExceedsDevices {
        need: u64,
        largest: u64,
    },
    CodeExceedsDevices {
        kernel: &'static str,
        need: u64,
        largest: u64,
    },
    /// `bytes` must move from `from` to `to`, and every path has an
    /// intermediate whose free buffer is smaller than the message.
    RouteUnavailable {
        from: usize,
        to: usize,
        bytes: u64,
    },
}

impl Cluster {
    pub fn devices(devices: Vec<Device>) -> Self {
        let links = derive_links(&devices);
        Self { devices, links }
    }

    /// Stage on an explicit fabric. The compiler still chooses the routes.
    pub fn linked(devices: Vec<Device>, links: Vec<Link>) -> Self {
        Self { devices, links }
    }
}

/// The fabric implied by a device list.
pub fn derive_links(devices: &[Device]) -> Vec<Link> {
    let mut cpus = Vec::new();
    let mut gpus = Vec::new();
    for (i, device) in devices.iter().enumerate() {
        if device.kind == DeviceKind::Cpu {
            cpus.push(i);
        } else {
            gpus.push(i);
        }
    }
    let mut links = Vec::new();
    connect_all(&mut links, &cpus);
    if cpus.is_empty() {
        connect_all(&mut links, &gpus);
    } else {
        let mut by_host = vec![Vec::new(); cpus.len()];
        for (i, &gpu) in gpus.iter().enumerate() {
            let host = i % cpus.len();
            links.push(Link {
                a: gpu,
                b: cpus[host],
            });
            by_host[host].push(gpu);
        }
        for group in &by_host {
            connect_all(&mut links, group);
        }
    }
    links
}

fn connect_all(links: &mut Vec<Link>, group: &[usize]) {
    for i in 0..group.len() {
        for j in (i + 1)..group.len() {
            links.push(Link {
                a: group[i],
                b: group[j],
            });
        }
    }
}

/// Shortest path whose intermediate devices have `free` bytes available.
pub fn route_message(links: &[Link], free: &[u64], from: usize, to: usize, bytes: u64) -> Result<Vec<usize>, StageError> {
    if from == to || bytes == 0 {
        return Ok(vec![from]);
    }
    let n = free.len();
    if from >= n || to >= n {
        return Err(StageError::RouteUnavailable { from, to, bytes });
    }
    let mut neighbors = vec![Vec::new(); n];
    for link in links {
        if link.a < n && link.b < n && link.a != link.b {
            neighbors[link.a].push(link.b);
            neighbors[link.b].push(link.a);
        }
    }
    let mut prev = vec![None; n];
    let mut queue = VecDeque::new();
    prev[from] = Some(from);
    queue.push_back(from);
    while let Some(node) = queue.pop_front() {
        if node == to {
            break;
        }
        for &next in &neighbors[node] {
            if prev[next].is_some() {
                continue;
            }
            if next != to && free[next] < bytes {
                continue;
            }
            prev[next] = Some(node);
            queue.push_back(next);
        }
    }
    let Some(_) = prev[to] else {
        return Err(StageError::RouteUnavailable { from, to, bytes });
    };
    let mut path = vec![to];
    let mut cursor = to;
    while cursor != from {
        cursor = prev[cursor].unwrap();
        path.push(cursor);
        if path.len() > n {
            return Err(StageError::RouteUnavailable { from, to, bytes });
        }
    }
    path.reverse();
    Ok(path)
}

/// Stage `cfg` at `batch` by `seq` and record the kernels each GPU must load.
pub fn compile_mixtral(
    cfg: &MixtralConfig,
    batch: u64,
    seq: u64,
    cluster: &Cluster,
) -> Result<Compiled, StageError> {
    let plan = stage_mixtral(cfg, batch, seq, cluster)?;
    Ok(finish(plan))
}

/// Stage a [`Program`]. Its tape stays resident for the whole schedule
/// because a skip reads an earlier save.
pub fn compile_program(prog: &Program, cluster: &Cluster) -> Result<Compiled, StageError> {
    let plan = stage_program(prog, cluster)?;
    Ok(finish(plan))
}

fn finish(plan: StagePlan) -> Compiled {
    let mut gpu_kernels = Vec::new();
    for window in &plan.windows {
        for phase in &window.code_phases {
            for kernel in phase {
                if *kernel != Kernel::CpuRuntime && !gpu_kernels.contains(kernel) {
                    gpu_kernels.push(*kernel);
                }
            }
        }
    }
    Compiled { plan, gpu_kernels }
}

pub fn stage_mixtral(
    cfg: &MixtralConfig,
    batch: u64,
    seq: u64,
    cluster: &Cluster,
) -> Result<StagePlan, StageError> {
    if cluster.devices.is_empty() {
        return Err(StageError::NoDevice);
    }
    let n = mul(batch, seq);
    let mut matrices = mixtral_matrices(cfg, n);
    let param_bytes = f32_bytes(matrices.iter().map(|m| mul(m.rows, m.cols)).sum());
    let mut planner = Planner::new(cluster, param_bytes)?;
    for matrix in &matrices {
        planner.place_param(matrix)?;
    }
    bind_offsets(cfg, &mut matrices);
    for layer in 0..cfg.layers {
        planner.place_tape(&format!("layer-{layer}"), layer_tape_bytes(cfg, n, seq)?)?;
    }
    planner.emit_embed(matrices.iter().find(|m| m.name == "embed").expect("embed"), n)?;
    let hd = cfg.head_dim as u64;
    for layer in 0..cfg.layers {
        for matrix in matrices.iter().filter(|m| m.layer == Some(layer)) {
            let name = matrix.name.clone();
            planner.emit_matrix(matrix)?;
            if name.ends_with("-v") {
                planner.emit_rope(layer as u32, true, n, cfg.q_dim() as u64)?;
                planner.emit_rope(layer as u32, false, n, cfg.kv_dim() as u64)?;
                planner.emit_attention(
                    &format!("L{layer}-gqa"),
                    n,
                    seq,
                    hd,
                    &[Kernel::GqaFwd],
                    &[Kernel::GqaBwd],
                )?;
            } else if name.ends_with("-o") {
                planner.emit_residual(layer as u32, true, n, cfg.dim as u64)?;
            } else if name.ends_with("-router") {
                planner.emit_router(layer as u32, n, cfg.dim as u64, cfg.experts as u64)?;
            } else if let Some(expert) = name.rsplit('-').next().and_then(|s| {
                name.contains("-up-").then(|| s.parse::<u32>().ok()).flatten()
            }) {
                planner.emit_silu(layer as u32, expert, n, cfg.intermediate as u64)?;
            }
        }
        planner.emit_mix(layer as u32, n, cfg.dim as u64, cfg.experts as u64)?;
        planner.emit_residual(layer as u32, false, n, cfg.dim as u64)?;
    }
    planner.emit_named(&matrices, "final-norm")?;
    planner.emit_named(&matrices, "lm-head")?;
    planner.emit_loss(n, cfg.vocab as u64)?;
    planner.emit_sgd(&matrices)?;
    while !planner.tape_marks.is_empty() {
        planner.release_tapes();
    }
    Ok(planner.flush())
}

pub fn stage_program(prog: &Program, cluster: &Cluster) -> Result<StagePlan, StageError> {
    if cluster.devices.is_empty() {
        return Err(StageError::NoDevice);
    }
    let matrices = program_matrices(prog);
    let param_bytes = f32_bytes(matrices.iter().map(|m| mul(m.rows, m.cols)).sum());
    let tape_bytes = prog.tape_f32 as u64 * 4 + prog.tape_u8 as u64;
    let mut planner = Planner::new(cluster, param_bytes + tape_bytes)?;
    for matrix in &matrices {
        planner.place_param(matrix)?;
    }
    planner.place_tape("program-tape", tape_bytes)?;
    for matrix in &matrices {
        planner.emit_matrix(matrix)?;
    }
    for stage in &prog.stages {
        if let Stage::Attention {
            batch, tokens, dim, ..
        } = *stage
        {
            planner.emit_attention(
                "attention",
                mul(batch as u64, tokens as u64),
                tokens as u64,
                dim as u64,
                &[Kernel::AttnFwd],
                &[Kernel::AttnBwd],
            )?;
        }
    }
    Ok(planner.flush())
}

struct Matrix {
    name: String,
    layer: Option<usize>,
    rows: u64,
    cols: u64,
    batch: u64,
    extra: u64,
    offset: u64,
    fwd: Vec<Kernel>,
    bwd: Vec<Kernel>,
    mask_per_col: bool,
}

struct Planner<'a> {
    cluster: &'a Cluster,
    free: Vec<u64>,
    home_bytes: Vec<u64>,
    peak: Vec<u64>,
    homes: Vec<Home>,
    tape_marks: Vec<usize>,
    windows: Vec<Window>,
    later: Vec<Window>,
    tail: Vec<Window>,
    transfers: Vec<Transfer>,
}

impl<'a> Planner<'a> {
    fn new(cluster: &'a Cluster, persistent: u64) -> Result<Self, StageError> {
        let capacity = cluster.devices.iter().map(|d| d.buffer_bytes).sum();
        if persistent > capacity {
            return Err(StageError::ParamsExceedCluster {
                params: persistent,
                capacity,
            });
        }
        let n = cluster.devices.len();
        Ok(Self {
            cluster,
            free: cluster.devices.iter().map(|d| d.buffer_bytes).collect(),
            home_bytes: vec![0; n],
            peak: vec![0; n],
            homes: Vec::new(),
            tape_marks: Vec::new(),
            windows: Vec::new(),
            later: Vec::new(),
            tail: Vec::new(),
            transfers: Vec::new(),
        })
    }

    fn push_pass(&mut self, window: Window) {
        if window.pass == Pass::Fwd {
            self.windows.push(window);
        } else {
            self.later.push(window);
        }
    }

    fn push_annotated(&mut self, window: Window) -> Result<(), StageError> {
        let window = self.annotate(window)?;
        self.push_pass(window);
        Ok(())
    }

    fn annotate(&mut self, mut window: Window) -> Result<Window, StageError> {
        let inbound = self.messages(&window, true)?;
        let outbound = self.messages(&window, false)?;
        for hop in inbound.iter().chain(&outbound) {
            self.transfers.push(Transfer {
                from: hop.from,
                to: hop.to,
                bytes: hop.bytes,
            });
        }
        window.inbound = inbound;
        window.outbound = outbound;
        Ok(window)
    }

    fn messages(&self, window: &Window, inbound: bool) -> Result<Vec<Hop>, StageError> {
        let mut hops = Vec::new();
        if window.param_bytes > 0 && window.home != window.device {
            if inbound {
                let path = self.route(window.home, window.device, window.param_bytes)?;
                hops.extend(expand_path(&path, HopKind::Param, window));
                if window.exec == Exec::Sgd {
                    let adjoint = self.route(window.home, window.device, window.param_bytes)?;
                    hops.extend(expand_path(&adjoint, HopKind::Adjoint, window));
                }
            } else if window.pass == Pass::Bwd {
                let path = self.route(window.device, window.home, window.param_bytes)?;
                hops.extend(expand_path(&path, HopKind::Adjoint, window));
            } else if window.exec == Exec::Sgd {
                let path = self.route(window.device, window.home, window.param_bytes)?;
                hops.extend(expand_path(&path, HopKind::Param, window));
            }
        }
        if window.tape_bytes > 0 {
            if let Some(act) = self.activation_home_for(window.tape_bytes) {
                let (from, to) = if inbound {
                    (act, window.device)
                } else {
                    (window.device, act)
                };
                if from != to {
                    let path = self.route(from, to, window.tape_bytes)?;
                    hops.extend(expand_path(&path, HopKind::Activation, window));
                }
            }
        }
        Ok(hops)
    }

    fn activation_home_for(&self, bytes: u64) -> Option<usize> {
        if bytes == 0 {
            return None;
        }
        self.cluster
            .devices
            .iter()
            .enumerate()
            .filter(|(i, _)| self.free[*i] >= bytes)
            .max_by_key(|(i, device)| (self.free[*i], device.kind == DeviceKind::Cpu))
            .map(|(i, _)| i)
    }

    fn activation_route(&self, device: usize, bytes: u64) -> Result<(), StageError> {
        let Some(home) = self.activation_home_for(bytes) else {
            return Ok(());
        };
        if home == device {
            return Ok(());
        }
        self.route(home, device, bytes).map(|_| ())
    }

    fn route(&self, from: usize, to: usize, bytes: u64) -> Result<Vec<usize>, StageError> {
        route_message(&self.cluster.links, &self.free, from, to, bytes)
    }

    fn sgd_device(&self, home: usize, bytes: u64) -> Result<usize, StageError> {
        let runnable = |device: usize| {
            code_phases(&self.cluster.devices[device], &[Kernel::Sgd]).is_some()
                && (device == home || self.route(home, device, bytes).is_ok())
        };
        if runnable(home) {
            return Ok(home);
        }
        self.cluster
            .devices
            .iter()
            .enumerate()
            .find(|(i, _)| runnable(*i))
            .map(|(i, _)| i)
            .ok_or(StageError::CodeExceedsDevices {
                kernel: "k_sgd",
                need: KERNEL_CODE_BYTES,
                largest: self
                    .cluster
                    .devices
                    .iter()
                    .map(|device| device.code_bytes)
                    .max()
                    .unwrap_or(0),
            })
    }

    fn flush(mut self) -> StagePlan {
        self.later.reverse();
        self.windows.append(&mut self.later);
        self.windows.append(&mut self.tail);
        self.finish()
    }

    fn place_param(&mut self, matrix: &Matrix) -> Result<(), StageError> {
        self.place_rect(HomeKind::Param, &matrix.name, matrix.rows, matrix.cols, 4)
    }

    fn place_tape(&mut self, name: &str, bytes: u64) -> Result<(), StageError> {
        let mark = self.homes.len();
        if bytes == 0 {
            self.tape_marks.push(mark);
            return Ok(());
        }
        self.place_rect(HomeKind::Tape, name, 1, bytes, 1)?;
        self.tape_marks.push(mark);
        Ok(())
    }

    fn release_tapes(&mut self) {
        let mark = self.tape_marks.pop().unwrap_or(self.homes.len());
        while self.homes.len() > mark {
            let home = self.homes.pop().expect("tape home");
            if home.kind == HomeKind::Tape {
                let bytes = home.bytes();
                self.free[home.device] = self.free[home.device].saturating_add(bytes);
                self.home_bytes[home.device] = self.home_bytes[home.device].saturating_sub(bytes);
            }
        }
    }

    fn place_rect(
        &mut self,
        kind: HomeKind,
        name: &str,
        rows: u64,
        cols: u64,
        elem: u64,
    ) -> Result<(), StageError> {
        let mut row = 0;
        while row < rows {
            let row_bytes = mul(cols, elem);
            if let Some(dev) = self.pick(row_bytes) {
                let fit = (self.free[dev] / row_bytes).min(rows - row).max(1);
                let take = fit.min(rows - row);
                let bytes = mul(mul(take, cols), elem);
                self.claim(dev, bytes);
                self.homes.push(Home {
                    kind,
                    name: name.to_string(),
                    device: dev,
                    row_begin: row,
                    row_end: row + take,
                    col_begin: 0,
                    col_end: cols,
                    stride: cols,
                });
                row += take;
                continue;
            }
            let mut col = 0;
            while col < cols {
                let dev = self.pick(elem).ok_or(StageError::TapeExceedsCluster {
                    tape: mul(cols, elem),
                    free: self.free.iter().copied().max().unwrap_or(0),
                })?;
                let take = (self.free[dev] / elem).min(cols - col);
                if take == 0 {
                    return Err(StageError::TapeExceedsCluster {
                        tape: elem,
                        free: self.free[dev],
                    });
                }
                let bytes = mul(take, elem);
                self.claim(dev, bytes);
                self.homes.push(Home {
                    kind,
                    name: name.to_string(),
                    device: dev,
                    row_begin: row,
                    row_end: row + 1,
                    col_begin: col,
                    col_end: col + take,
                    stride: cols,
                });
                col += take;
            }
            row += 1;
        }
        Ok(())
    }

    fn claim(&mut self, dev: usize, bytes: u64) {
        self.free[dev] -= bytes;
        self.home_bytes[dev] += bytes;
        self.peak[dev] = self.peak[dev].max(self.home_bytes[dev]);
    }

    fn pick(&self, need: u64) -> Option<usize> {
        let cpu = self
            .cluster
            .devices
            .iter()
            .enumerate()
            .filter(|(i, d)| d.kind == DeviceKind::Cpu && self.free[*i] >= need)
            .max_by_key(|(i, _)| self.free[*i]);
        if let Some((i, _)) = cpu {
            return Some(i);
        }
        self.cluster
            .devices
            .iter()
            .enumerate()
            .filter(|(i, _)| self.free[*i] >= need)
            .max_by_key(|(i, _)| self.free[*i])
            .map(|(i, _)| i)
    }

    fn emit_named(&mut self, matrices: &[Matrix], name: &str) -> Result<(), StageError> {
        let matrix = matrices
            .iter()
            .find(|m| m.name == name)
            .expect("named matrix");
        self.emit_matrix(matrix)
    }

    fn emit_rope(&mut self, layer: u32, query: bool, n: u64, width: u64) -> Result<(), StageError> {
        let exec = if query { Exec::RopeQ } else { Exec::RopeK };
        self.emit_simple(
            &format!("L{layer}-{}", if query { "rope-q" } else { "rope-k" }),
            exec,
            &[Kernel::Rope],
            &[Kernel::Rope],
            f32_bytes(mul(n, width) * 2),
            layer,
            0,
            0,
            n,
            width,
        )
    }

    fn emit_residual(&mut self, layer: u32, attn: bool, n: u64, dim: u64) -> Result<(), StageError> {
        self.emit_simple(
            &format!("L{layer}-{}", if attn { "add-attn" } else { "add-moe" }),
            if attn { Exec::AddAttn } else { Exec::AddMoe },
            &[Kernel::Add],
            &[Kernel::Add],
            f32_bytes(mul(n, dim) * 2),
            layer,
            0,
            0,
            n,
            dim,
        )
    }

    fn emit_router(&mut self, layer: u32, n: u64, dim: u64, experts: u64) -> Result<(), StageError> {
        self.emit_simple(
            &format!("L{layer}-router-gate"),
            Exec::Router,
            &[Kernel::Router],
            &[Kernel::RouterBwd],
            f32_bytes(mul(n, dim) + mul(n, experts) * 4),
            layer,
            0,
            0,
            n,
            experts,
        )
    }

    fn emit_silu(&mut self, layer: u32, expert: u32, n: u64, inter: u64) -> Result<(), StageError> {
        self.emit_simple(
            &format!("L{layer}-silu-{expert}"),
            Exec::Silu,
            &[Kernel::Silu],
            &[Kernel::SiluBwd],
            f32_bytes(mul(n, inter) * 4),
            layer,
            expert,
            0,
            n,
            inter,
        )
    }

    fn emit_mix(&mut self, layer: u32, n: u64, dim: u64, experts: u64) -> Result<(), StageError> {
        self.emit_simple(
            &format!("L{layer}-mix"),
            Exec::Mix,
            &[Kernel::Add],
            &[Kernel::Add],
            f32_bytes(mul(n, dim) * (experts + 2)),
            layer,
            0,
            0,
            n,
            dim,
        )
    }

    fn emit_simple(
        &mut self,
        name: &str,
        exec: Exec,
        fwd: &[Kernel],
        bwd: &[Kernel],
        footprint: u64,
        layer: u32,
        expert: u32,
        batch_begin: u64,
        batch_end: u64,
        width: u64,
    ) -> Result<(), StageError> {
        let mut kernels = fwd.to_vec();
        kernels.extend_from_slice(bwd);
        let (device, resident, _) = self.place_window(footprint, usize::MAX, 0, footprint, &kernels)?;
        for (pass, ks) in [(Pass::Fwd, fwd), (Pass::Bwd, bwd)] {
            let phases =
                code_phases(&self.cluster.devices[device], ks).expect("device accepted the op");
            self.push_annotated(Window {
                name: name.to_string(),
                device,
                code_phases: phases,
                resident_bytes: resident,
                tape_bytes: footprint,
                param_bytes: 0,
                pass,
                exec,
                param_offset: 0,
                batch_begin,
                batch_end,
                row_begin: 0,
                row_end: width,
                col_begin: 0,
                col_end: width,
                din: width,
                dout: width,
                home: device,
                layer,
                expert,
                inbound: Vec::new(),
                outbound: Vec::new(),
            })?;
        }
        Ok(())
    }

    fn emit_embed(&mut self, matrix: &Matrix, n: u64) -> Result<(), StageError> {
        let table = f32_bytes(mul(matrix.rows, matrix.cols));
        let activation = f32_bytes(mul(n, matrix.cols));
        let footprint = table + activation;
        let kernels = [Kernel::Embed, Kernel::EmbedBwd];
        let home = self.home_device(&matrix.name, 0, 0);
        let (device, resident, _) =
            self.place_window(footprint, home, table, activation, &kernels)?;
        for (pass, ks) in [
            (Pass::Fwd, &[Kernel::Embed][..]),
            (Pass::Bwd, &[Kernel::EmbedBwd][..]),
        ] {
            let phases = code_phases(&self.cluster.devices[device], ks).expect("embed device");
            self.push_annotated(Window {
                name: format!("embed[0:{n}]"),
                device,
                code_phases: phases,
                resident_bytes: resident,
                tape_bytes: f32_bytes(mul(n, matrix.cols)),
                param_bytes: table,
                pass,
                exec: Exec::Embed,
                param_offset: matrix.offset,
                batch_begin: 0,
                batch_end: n,
                row_begin: 0,
                row_end: matrix.rows,
                col_begin: 0,
                col_end: matrix.cols,
                din: matrix.rows,
                dout: matrix.cols,
                home,
                layer: 0,
                expert: 0,
                inbound: Vec::new(),
                outbound: Vec::new(),
            })?;
        }
        Ok(())
    }

    fn emit_sgd(&mut self, matrices: &[Matrix]) -> Result<(), StageError> {
        let homes = self.homes.clone();
        let mut placed = false;
        for home in homes.iter().filter(|h| h.kind == HomeKind::Param) {
            let matrix = matrices
                .iter()
                .find(|m| m.name == home.name)
                .expect("parameter home");
            let bytes = home.bytes();
            let device = self.sgd_device(home.device, bytes)?;
            let phases = code_phases(&self.cluster.devices[device], &[Kernel::Sgd])
                .expect("sgd device accepted the update");
            placed = true;
            let window = self.annotate(Window {
                name: format!(
                    "sgd-{}[{}:{},{}:{}]",
                    home.name, home.row_begin, home.row_end, home.col_begin, home.col_end
                ),
                device,
                code_phases: phases,
                resident_bytes: bytes,
                tape_bytes: 0,
                param_bytes: bytes,
                pass: Pass::Fwd,
                exec: Exec::Sgd,
                param_offset: matrix.offset,
                batch_begin: 0,
                batch_end: 0,
                row_begin: home.row_begin,
                row_end: home.row_end,
                col_begin: home.col_begin,
                col_end: home.col_end,
                din: matrix.rows,
                dout: matrix.cols,
                home: home.device,
                layer: matrix.layer.unwrap_or(0) as u32,
                expert: matrix_expert(&matrix.name),
                inbound: Vec::new(),
                outbound: Vec::new(),
            })?;
            self.tail.push(window);
        }
        if placed {
            Ok(())
        } else {
            Err(StageError::CodeExceedsDevices {
                kernel: "k_sgd",
                need: KERNEL_CODE_BYTES,
                largest: self
                    .cluster
                    .devices
                    .iter()
                    .map(|d| d.code_bytes)
                    .max()
                    .unwrap_or(0),
            })
        }
    }

    fn emit_matrix(&mut self, matrix: &Matrix) -> Result<(), StageError> {
        let (row_tile, din_tile, col_tile) = self.fit_linear(
            matrix.batch,
            matrix.rows,
            matrix.cols,
            matrix.mask_per_col,
            matrix.extra,
        )?;
        let mut batch = 0;
        while batch < matrix.batch {
            let batch_end = (batch + row_tile).min(matrix.batch);
            let mut din = 0;
            while din < matrix.rows {
                let din_end = (din + din_tile).min(matrix.rows);
                let mut col = 0;
                while col < matrix.cols {
                    let col_end = (col + col_tile).min(matrix.cols);
                    self.emit_linear_tile(matrix, batch, batch_end, din, din_end, col, col_end)?;
                    col = col_end;
                }
                din = din_end;
            }
            batch = batch_end;
        }
        Ok(())
    }

    fn emit_linear_tile(
        &mut self,
        matrix: &Matrix,
        batch: u64,
        batch_end: u64,
        din: u64,
        din_end: u64,
        col: u64,
        col_end: u64,
    ) -> Result<(), StageError> {
        let rows = batch_end - batch;
        let k = din_end - din;
        let c = col_end - col;
        let param = f32_bytes(mul(k, c));
        let tape = f32_bytes(mul(rows, k))
            + f32_bytes(mul(rows, c))
            + if matrix.mask_per_col { mul(rows, c) } else { 0 };
        let footprint = linear_footprint(rows, k, c, matrix.mask_per_col) + matrix.extra;
        let kernels: Vec<Kernel> = matrix
            .fwd
            .iter()
            .chain(matrix.bwd.iter())
            .copied()
            .collect();
        let home_device = self.home_device(&matrix.name, din, col);
        let (device, resident, _) =
            self.place_window(footprint, home_device, param, tape, &kernels)?;
        let exec = exec_of(matrix.fwd.first().copied().unwrap_or(Kernel::Linear));
        let layer = matrix.layer.unwrap_or(0) as u32;
        let expert = matrix_expert(&matrix.name);
        for pass in [Pass::Fwd, Pass::Bwd] {
            let kernels = if pass == Pass::Fwd {
                &matrix.fwd
            } else {
                &matrix.bwd
            };
            let dev = &self.cluster.devices[device];
            let phases = code_phases(dev, kernels).expect("device accepted these kernels");
            self.push_annotated(Window {
                name: format!(
                    "{}[{batch}:{batch_end},{din}:{din_end},{col}:{col_end}]",
                    matrix.name
                ),
                device,
                code_phases: phases,
                resident_bytes: resident,
                tape_bytes: tape,
                param_bytes: param,
                pass,
                exec,
                param_offset: matrix.offset,
                batch_begin: batch,
                batch_end,
                row_begin: din,
                row_end: din_end,
                col_begin: col,
                col_end,
                din: matrix.rows,
                dout: matrix.cols,
                home: home_device,
                layer,
                expert,
                inbound: Vec::new(),
                outbound: Vec::new(),
            })?;
        }
        Ok(())
    }

    fn emit_attention(
        &mut self,
        name: &str,
        queries: u64,
        keys: u64,
        dim: u64,
        fwd: &[Kernel],
        bwd: &[Kernel],
    ) -> Result<(), StageError> {
        let (q_tile, k_tile, feat) = self.fit_attention(queries, keys, dim, fwd, bwd)?;
        let mut q = 0;
        while q < queries {
            let q_end = (q + q_tile).min(queries);
            let mut feature = 0;
            while feature < dim {
                let f_end = (feature + feat).min(dim);
                let mut k = 0;
                while k < keys {
                    let k_end = (k + k_tile).min(keys);
                    let qn = q_end - q;
                    let kn = k_end - k;
                    let fnn = f_end - feature;
                    let tape = f32_bytes(mul(qn, kn));
                    let footprint = attn_footprint(qn, kn, fnn);
                    let kernels: Vec<Kernel> = fwd.iter().chain(bwd.iter()).copied().collect();
                    let (device, resident, _) =
                        self.place_window(footprint, usize::MAX, 0, tape, &kernels)?;
                    let layer = name
                        .trim_start_matches('L')
                        .split('-')
                        .next()
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(0);
                    for (pass, ks) in [(Pass::Fwd, fwd), (Pass::Bwd, bwd)] {
                        let phases = code_phases(&self.cluster.devices[device], ks)
                            .expect("device accepted attention");
                        self.push_annotated(Window {
                            name: format!("{name}[{q}:{q_end},{k}:{k_end},{feature}:{f_end}]"),
                            device,
                            code_phases: phases,
                            resident_bytes: resident,
                            tape_bytes: tape,
                            param_bytes: 0,
                            pass,
                            exec: Exec::Gqa,
                            param_offset: 0,
                            batch_begin: q,
                            batch_end: q_end,
                            row_begin: k,
                            row_end: k_end,
                            col_begin: feature,
                            col_end: f_end,
                            din: keys,
                            dout: dim,
                            home: device,
                            layer,
                            expert: 0,
                            inbound: Vec::new(),
                            outbound: Vec::new(),
                        })?;
                    }
                    k = k_end;
                }
                feature = f_end;
            }
            q = q_end;
        }
        Ok(())
    }

    fn emit_loss(&mut self, tokens: u64, vocab: u64) -> Result<(), StageError> {
        // The loss tile stores logits. The contraction width is 1 and the
        // columns are the vocabulary.
        let (row_tile, _, col_tile) = self.fit_linear(tokens, 1, vocab, false, 0)?;
        let mut row = 0;
        while row < tokens {
            let row_end = (row + row_tile).min(tokens);
            let mut col = 0;
            while col < vocab {
                let col_end = (col + col_tile).min(vocab);
                let rows = row_end - row;
                let c = col_end - col;
                let tape = f32_bytes(mul(rows, c));
                let footprint = linear_footprint(rows, 1, c, false);
                let kernels = [Kernel::Ce, Kernel::Sgd];
                let (device, resident, _) =
                    self.place_window(footprint, usize::MAX, 0, tape, &kernels)?;
                for (pass, ks) in [
                    (Pass::Fwd, &[Kernel::Ce][..]),
                    (Pass::Bwd, &[Kernel::Ce][..]),
                ] {
                    let phases = code_phases(&self.cluster.devices[device], ks)
                        .expect("device accepted the loss");
                    self.push_annotated(Window {
                        name: format!("loss[{row}:{row_end},{col}:{col_end}]"),
                        device,
                        code_phases: phases,
                        resident_bytes: resident,
                        tape_bytes: tape,
                        param_bytes: 0,
                        pass,
                        exec: Exec::Loss,
                        param_offset: 0,
                        batch_begin: row,
                        batch_end: row_end,
                        row_begin: 0,
                        row_end: 1,
                        col_begin: col,
                        col_end,
                        din: 1,
                        dout: vocab,
                        home: device,
                        layer: 0,
                        expert: 0,
                        inbound: Vec::new(),
                        outbound: Vec::new(),
                    })?;
                }
                col = col_end;
            }
            row = row_end;
        }
        Ok(())
    }

    fn fit_linear(
        &self,
        batch: u64,
        din: u64,
        dout: u64,
        mask: bool,
        extra: u64,
    ) -> Result<(u64, u64, u64), StageError> {
        let kernels = [Kernel::Linear, Kernel::LinearDx, Kernel::LinearDw];
        self.shrink3(
            batch,
            din,
            dout,
            |r, k, c| linear_footprint(r, k, c, mask) + extra,
            &kernels,
        )
    }

    fn fit_attention(
        &self,
        queries: u64,
        keys: u64,
        dim: u64,
        fwd: &[Kernel],
        bwd: &[Kernel],
    ) -> Result<(u64, u64, u64), StageError> {
        let kernels: Vec<Kernel> = fwd.iter().chain(bwd).copied().collect();
        self.shrink3(queries, keys, dim, attn_footprint, &kernels)
    }

    fn shrink3(
        &self,
        mut a: u64,
        mut b: u64,
        mut c: u64,
        footprint: impl Fn(u64, u64, u64) -> u64,
        kernels: &[Kernel],
    ) -> Result<(u64, u64, u64), StageError> {
        let full_a = a.max(1);
        let full_b = b.max(1);
        let full_c = c.max(1);
        a = full_a;
        b = full_b;
        c = full_c;
        loop {
            let need = footprint(a, b, c);
            if self.some_device_fits(need, kernels) {
                return Ok((a, b, c));
            }
            if c >= a && c >= b && c > 1 {
                c = (c + 1) / 2;
            } else if a >= b && a > 1 {
                a = (a + 1) / 2;
            } else if b > 1 {
                b = (b + 1) / 2;
            } else if self.memory_and_code_fit(need, kernels) {
                return Err(self.blocked_route(need));
            } else {
                let largest = self
                    .cluster
                    .devices
                    .iter()
                    .map(|d| d.buffer_bytes)
                    .max()
                    .unwrap_or(0);
                return Err(StageError::TileExceedsDevices { need, largest });
            }
        }
    }

    fn memory_and_code_fit(&self, footprint: u64, kernels: &[Kernel]) -> bool {
        self.cluster.devices.iter().enumerate().any(|(i, dev)| {
            self.resident(i, footprint, usize::MAX, 0) <= dev.buffer_bytes
                && code_phases(dev, kernels).is_some()
        })
    }

    fn some_device_fits(&self, footprint: u64, kernels: &[Kernel]) -> bool {
        self.cluster.devices.iter().enumerate().any(|(i, dev)| {
            self.resident(i, footprint, usize::MAX, 0) <= dev.buffer_bytes
                && code_phases(dev, kernels).is_some()
                && self.activation_route(i, footprint).is_ok()
        })
    }

    fn blocked_route(&self, bytes: u64) -> StageError {
        let from = self.activation_home_for(bytes).unwrap_or(0);
        let to = self
            .cluster
            .devices
            .iter()
            .enumerate()
            .find(|(i, dev)| {
                self.resident(*i, bytes, usize::MAX, 0) <= dev.buffer_bytes && *i != from
            })
            .map(|(i, _)| i)
            .unwrap_or(from);
        StageError::RouteUnavailable { from, to, bytes }
    }

    fn place_window(
        &mut self,
        footprint: u64,
        home_device: usize,
        param_on_home: u64,
        activation: u64,
        kernels: &[Kernel],
    ) -> Result<(usize, u64, Vec<Vec<Kernel>>), StageError> {
        let mut best: Option<(usize, u64, Vec<Vec<Kernel>>)> = None;
        let mut blocked: Option<StageError> = None;
        for (i, dev) in self.cluster.devices.iter().enumerate() {
            let Some(phases) = code_phases(dev, kernels) else {
                continue;
            };
            let resident = self.resident(i, footprint, home_device, param_on_home);
            if resident > dev.buffer_bytes {
                continue;
            }
            if param_on_home > 0 && home_device != usize::MAX && home_device != i {
                if let Err(err) = self.route(home_device, i, param_on_home) {
                    blocked = Some(err);
                    continue;
                }
            }
            if let Err(err) = self.activation_route(i, activation) {
                blocked = Some(err);
                continue;
            }
            let prefer_gpu = dev.kind == DeviceKind::Gpu;
            let replace = match &best {
                None => true,
                Some((j, _, _)) => {
                    prefer_gpu && self.cluster.devices[*j].kind != DeviceKind::Gpu
                        || (prefer_gpu == (self.cluster.devices[*j].kind == DeviceKind::Gpu)
                            && resident < self.resident(*j, footprint, home_device, param_on_home))
                }
            };
            if replace {
                best = Some((i, resident, phases));
            }
        }
        let (device, resident, phases) = best.ok_or_else(|| {
            if let Some(err) = blocked {
                return err;
            }
            let largest_code = self
                .cluster
                .devices
                .iter()
                .map(|d| d.code_bytes)
                .max()
                .unwrap_or(0);
            if let Some(kernel) = kernels
                .iter()
                .find(|k| self.cluster.devices.iter().all(|d| !kernel_fits(d, **k)))
            {
                StageError::CodeExceedsDevices {
                    kernel: kernel.name(),
                    need: kernel.code_bytes(),
                    largest: largest_code,
                }
            } else {
                StageError::TileExceedsDevices {
                    need: footprint,
                    largest: self.free.iter().copied().max().unwrap_or(0),
                }
            }
        })?;
        self.peak[device] = self.peak[device].max(resident);
        Ok((device, resident, phases))
    }

    fn resident(&self, device: usize, footprint: u64, home_device: usize, param: u64) -> u64 {
        // Homes stay allocated. The footprint includes a working copy of the
        // parameter tile; drop that copy when this device is the tile's home.
        let discount = if device == home_device { param } else { 0 };
        self.home_bytes[device] + footprint - discount
    }

    fn home_device(&self, name: &str, row: u64, col: u64) -> usize {
        self.homes
            .iter()
            .find(|h| {
                h.kind == HomeKind::Param
                    && h.name == name
                    && row >= h.row_begin
                    && row < h.row_end
                    && col >= h.col_begin
                    && col < h.col_end
            })
            .map(|h| h.device)
            .unwrap_or(0)
    }

    fn finish(self) -> StagePlan {
        StagePlan {
            homes: self.homes,
            windows: self.windows,
            transfers: self.transfers,
            peak_bytes: self.peak,
        }
    }
}

fn expand_path(path: &[usize], kind: HopKind, window: &Window) -> Vec<Hop> {
    let bytes = match kind {
        HopKind::Activation => window.tape_bytes,
        _ => window.param_bytes,
    };
    let elems = match kind {
        HopKind::Activation => bytes / 4,
        _ => window
            .row_end
            .saturating_sub(window.row_begin)
            .saturating_mul(window.col_end.saturating_sub(window.col_begin)),
    };
    path.windows(2)
        .map(|hop| Hop {
            from: hop[0],
            to: hop[1],
            bytes,
            kind,
            offset: window.param_offset,
            elems,
            row_begin: window.row_begin,
            row_end: window.row_end,
            col_begin: window.col_begin,
            col_end: window.col_end,
            stride: window.dout,
        })
        .collect()
}

fn code_phases(dev: &Device, kernels: &[Kernel]) -> Option<Vec<Vec<Kernel>>> {
    match dev.kind {
        DeviceKind::Cpu => {
            if dev.code_bytes >= CPU_RUNTIME_BYTES {
                Some(vec![vec![Kernel::CpuRuntime]])
            } else {
                None
            }
        }
        DeviceKind::Gpu => {
            let mut phases = Vec::new();
            let mut cur = Vec::new();
            let mut used = 0u64;
            for kernel in kernels {
                let bytes = kernel.code_bytes();
                if bytes > dev.code_bytes {
                    return None;
                }
                if used + bytes > dev.code_bytes && !cur.is_empty() {
                    phases.push(std::mem::take(&mut cur));
                    used = 0;
                }
                cur.push(*kernel);
                used += bytes;
            }
            if !cur.is_empty() {
                phases.push(cur);
            }
            Some(phases)
        }
    }
}

fn kernel_fits(dev: &Device, kernel: Kernel) -> bool {
    match dev.kind {
        DeviceKind::Cpu => dev.code_bytes >= CPU_RUNTIME_BYTES,
        DeviceKind::Gpu => dev.code_bytes >= kernel.code_bytes(),
    }
}

fn linear_footprint(rows: u64, din: u64, dout: u64, mask: bool) -> u64 {
    let x = mul(rows, din);
    let y = mul(rows, dout);
    let w = mul(din, dout);
    f32_bytes(x + x + y + w + w) + if mask { mul(rows, dout) } else { 0 }
}

fn attn_footprint(queries: u64, keys: u64, dim: u64) -> u64 {
    let q = mul(queries, dim);
    let k = mul(keys, dim);
    let probs = mul(queries, keys);
    f32_bytes(q + q + k + k + probs + probs)
}

fn f32_bytes(elems: u64) -> u64 {
    mul(elems, 4)
}

fn mul(a: u64, b: u64) -> u64 {
    a.checked_mul(b).expect("size exceeds u64")
}

fn matrix_expert(name: &str) -> u32 {
    if name.ends_with("ffn-norm") {
        1
    } else if name == "final-norm" {
        2
    } else {
        name.rsplit('-')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0)
    }
}

fn exec_of(kernel: Kernel) -> Exec {
    match kernel {
        Kernel::Embed | Kernel::EmbedBwd => Exec::Embed,
        Kernel::RmsNorm | Kernel::RmsNormBwd => Exec::RmsNorm,
        _ => Exec::Linear,
    }
}

fn bind_offsets(cfg: &MixtralConfig, matrices: &mut [Matrix]) {
    let lay = layout(cfg);
    for matrix in matrices.iter_mut() {
        matrix.offset = param_offset(cfg, &lay, &matrix.name) as u64;
    }
    let covered: u64 = matrices.iter().map(|m| mul(m.rows, m.cols)).sum();
    assert_eq!(
        covered,
        lay.total as u64,
        "staged matrices do not cover the parameter buffer"
    );
}

fn param_offset(cfg: &MixtralConfig, lay: &crate::mixtral::Layout, name: &str) -> usize {
    if name == "embed" {
        return lay.embed;
    }
    if name == "final-norm" {
        return lay.final_norm;
    }
    if name == "lm-head" {
        return lay.lm_head;
    }
    let (layer, rest) = name[1..].split_once('-').expect("layer matrix name");
    let layer: usize = layer.parse().expect("layer index");
    let ly = &lay.layers[layer];
    match rest {
        "attn-norm" => ly.attn_norm,
        "q" => ly.q,
        "k" => ly.k,
        "v" => ly.v,
        "o" => ly.o,
        "ffn-norm" => ly.ffn_norm,
        "router" => ly.router,
        other => {
            let (kind, index) = other.split_once('-').expect("expert matrix");
            let index: usize = index.parse().expect("expert index");
            match kind {
                "gate" => ly.gate[index],
                "up" => ly.up[index],
                "down" => ly.down[index],
                _ => panic!("unknown parameter {name}"),
            }
        }
    }
    .saturating_add(0 * cfg.dim)
}

fn mixtral_matrices(cfg: &MixtralConfig, n: u64) -> Vec<Matrix> {
    let dim = cfg.dim as u64;
    let mut embed = matrix(
        "embed",
        None,
        cfg.vocab as u64,
        dim,
        1,
        vec![Kernel::Embed],
        vec![Kernel::EmbedBwd],
        false,
    );
    embed.extra = f32_bytes(mul(n, dim));
    let mut out = vec![embed];
    for layer in 0..cfg.layers {
        out.extend(layer_matrices(cfg, layer, n));
    }
    out.push(matrix(
        "final-norm",
        None,
        dim,
        1,
        n,
        vec![Kernel::RmsNorm],
        vec![Kernel::RmsNormBwd],
        false,
    ));
    out.push(matrix(
        "lm-head",
        None,
        dim,
        cfg.vocab as u64,
        n,
        vec![Kernel::Linear],
        vec![Kernel::LinearDx, Kernel::LinearDw],
        false,
    ));
    out
}

fn layer_matrices(cfg: &MixtralConfig, layer: usize, n: u64) -> Vec<Matrix> {
    let dim = cfg.dim as u64;
    let q = cfg.q_dim() as u64;
    let kv = cfg.kv_dim() as u64;
    let inter = cfg.intermediate as u64;
    let experts = cfg.experts as u64;
    let tag = Some(layer);
    let mut out = vec![
        matrix(
            &format!("L{layer}-attn-norm"),
            tag,
            dim,
            1,
            n,
            vec![Kernel::RmsNorm],
            vec![Kernel::RmsNormBwd],
            false,
        ),
        matrix(
            &format!("L{layer}-q"),
            tag,
            dim,
            q,
            n,
            vec![Kernel::Linear],
            vec![Kernel::LinearDx, Kernel::LinearDw],
            false,
        ),
        matrix(
            &format!("L{layer}-k"),
            tag,
            dim,
            kv,
            n,
            vec![Kernel::Linear],
            vec![Kernel::LinearDx, Kernel::LinearDw],
            false,
        ),
        matrix(
            &format!("L{layer}-v"),
            tag,
            dim,
            kv,
            n,
            vec![Kernel::Linear],
            vec![Kernel::LinearDx, Kernel::LinearDw],
            false,
        ),
        matrix(
            &format!("L{layer}-o"),
            tag,
            q,
            dim,
            n,
            vec![Kernel::Linear],
            vec![Kernel::LinearDx, Kernel::LinearDw],
            false,
        ),
        matrix(
            &format!("L{layer}-ffn-norm"),
            tag,
            dim,
            1,
            n,
            vec![Kernel::RmsNorm],
            vec![Kernel::RmsNormBwd],
            false,
        ),
        matrix(
            &format!("L{layer}-router"),
            tag,
            dim,
            experts,
            n,
            vec![Kernel::Linear],
            vec![Kernel::LinearDx, Kernel::LinearDw],
            false,
        ),
    ];
    for expert in 0..cfg.experts {
        out.push(matrix(
            &format!("L{layer}-gate-{expert}"),
            tag,
            dim,
            inter,
            n,
            vec![Kernel::Linear],
            vec![Kernel::LinearDx, Kernel::LinearDw],
            false,
        ));
        out.push(matrix(
            &format!("L{layer}-up-{expert}"),
            tag,
            dim,
            inter,
            n,
            vec![Kernel::Linear],
            vec![Kernel::LinearDx, Kernel::LinearDw],
            false,
        ));
        out.push(matrix(
            &format!("L{layer}-down-{expert}"),
            tag,
            inter,
            dim,
            n,
            vec![Kernel::Linear],
            vec![Kernel::LinearDx, Kernel::LinearDw],
            false,
        ));
    }
    out
}

fn matrix(
    name: &str,
    layer: Option<usize>,
    rows: u64,
    cols: u64,
    batch: u64,
    fwd: Vec<Kernel>,
    bwd: Vec<Kernel>,
    mask_per_col: bool,
) -> Matrix {
    Matrix {
        name: name.to_string(),
        layer,
        rows,
        cols,
        batch,
        extra: 0,
        offset: 0,
        fwd,
        bwd,
        mask_per_col,
    }
}

fn layer_tape_bytes(cfg: &MixtralConfig, n: u64, seq: u64) -> Result<u64, StageError> {
    let dim = cfg.dim as u64;
    let q = cfg.q_dim() as u64;
    let kv = cfg.kv_dim() as u64;
    let heads = cfg.heads as u64;
    let experts = cfg.experts as u64;
    let inter = cfg.intermediate as u64;
    let topk = cfg.top_k as u64;
    let floats = mul(n, dim)
        + n
        + mul(n, q)
        + mul(n, kv)
        + mul(n, kv)
        + mul(mul(n, heads), seq)
        + mul(n, q)
        + mul(n, dim)
        + n
        + mul(n, experts)
        + mul(n, topk)
        + mul(n, topk)
        + mul(mul(experts, n), inter)
        + mul(mul(experts, n), inter);
    let index_bytes = mul(mul(n, topk), 8);
    Ok(f32_bytes(floats) + index_bytes)
}

fn program_matrices(prog: &Program) -> Vec<Matrix> {
    let mut out = Vec::new();
    for (i, stage) in prog.stages.iter().enumerate() {
        if let Stage::Gemm {
            batch,
            din,
            dout,
            relu,
            ..
        } = *stage
        {
            let mut bwd = vec![Kernel::GemmDx, Kernel::GemmDw, Kernel::GemmDb];
            if relu {
                bwd.insert(0, Kernel::ReluBwd);
            }
            out.push(matrix(
                &format!("gemm-{i}"),
                None,
                din as u64,
                dout as u64,
                batch as u64,
                vec![Kernel::GemmFwd],
                bwd,
                relu,
            ));
            out.push(matrix(
                &format!("bias-{i}"),
                None,
                1,
                dout as u64,
                batch as u64,
                vec![Kernel::GemmFwd],
                vec![Kernel::GemmDb],
                false,
            ));
        }
    }
    out
}

impl StagePlan {
    pub fn param_elems(&self) -> u64 {
        self.homes
            .iter()
            .filter(|h| h.kind == HomeKind::Param)
            .map(Home::elems)
            .sum()
    }

    pub fn check(&self, cluster: &Cluster, param_elems: u64) -> Result<(), String> {
        if self.param_elems() != param_elems {
            return Err(format!(
                "parameter homes cover {} elements, model has {param_elems}",
                self.param_elems()
            ));
        }
        for (i, dev) in cluster.devices.iter().enumerate() {
            let stored: u64 = self
                .homes
                .iter()
                .filter(|h| h.device == i)
                .map(Home::bytes)
                .sum();
            if stored > dev.buffer_bytes {
                return Err(format!(
                    "{} stores {stored} bytes with a buffer of {}",
                    dev.name, dev.buffer_bytes
                ));
            }
            if self.peak_bytes.get(i).copied().unwrap_or(0) > dev.buffer_bytes {
                return Err(format!(
                    "{} peaks at {} bytes with a buffer of {}",
                    dev.name, self.peak_bytes[i], dev.buffer_bytes
                ));
            }
        }
        for window in &self.windows {
            let dev = &cluster.devices[window.device];
            if window.resident_bytes > dev.buffer_bytes {
                return Err(format!(
                    "window {} uses {} bytes on {}",
                    window.name, window.resident_bytes, dev.name
                ));
            }
            for phase in &window.code_phases {
                let code: u64 = phase.iter().map(|k| k.code_bytes()).sum();
                if code > dev.code_bytes {
                    return Err(format!(
                        "window {} loads {code} code bytes on {}",
                        window.name, dev.name
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Program;
    use crate::mixtral::MixtralConfig;

    fn cpu(name: &str, buffer: u64, code: u64) -> Device {
        Device {
            name: name.into(),
            kind: DeviceKind::Cpu,
            buffer_bytes: buffer,
            code_bytes: code,
        }
    }

    fn gpu(name: &str, buffer: u64, code: u64) -> Device {
        Device {
            name: name.into(),
            kind: DeviceKind::Gpu,
            buffer_bytes: buffer,
            code_bytes: code,
        }
    }

    #[test]
    fn demo_stages_on_one_gpu() {
        let cfg = MixtralConfig::demo();
        let cluster = Cluster::devices(vec![gpu("gpu0", 64 << 20, 1 << 20)]);
        let compiled = compile_mixtral(&cfg, 2, 8, &cluster).expect("demo fits");
        compiled
            .plan
            .check(&cluster, crate::mixtral::parameter_count(&cfg))
            .unwrap();
        assert!(compiled.plan.windows.iter().all(|w| w.device == 0));
        assert!(compiled.gpu_kernels.contains(&Kernel::Linear));
        assert!(compiled.plan.windows.iter().any(|w| w.tape_bytes > 0));
    }

    #[test]
    fn published_mixtral_stages_across_cpu_storage_and_a_small_gpu() {
        let cfg = MixtralConfig::mixtral_8x7b();
        let params = crate::mixtral::parameter_count(&cfg);
        let cluster = Cluster::devices(vec![
            cpu("cpu0", 256 << 30, 1 << 20),
            gpu("gpu0", 6 << 30, KERNEL_CODE_BYTES),
        ]);
        let compiled = compile_mixtral(&cfg, 1, 1, &cluster).expect("published model stages");
        compiled.plan.check(&cluster, params).unwrap();
        assert!(
            compiled
                .plan
                .homes
                .iter()
                .filter(|h| h.kind == HomeKind::Param)
                .all(|h| h.device == 0),
            "parameter homes sit on the CPU"
        );
        assert!(
            compiled.plan.windows.iter().any(|w| w.device == 1),
            "compute uses the GPU"
        );
        assert!(
            compiled.plan.transfers.iter().any(|t| t.from != t.to),
            "tiles move between the CPU home and the GPU"
        );
        assert!(
            compiled
                .plan
                .windows
                .iter()
                .filter(|w| w.device == 1)
                .any(|w| w.code_phases.len() > 1),
            "the GPU code budget stages kernels into phases"
        );
        let tape: u64 = compiled
            .plan
            .homes
            .iter()
            .filter(|h| h.kind == HomeKind::Tape)
            .map(Home::bytes)
            .sum();
        assert_eq!(tape, 0, "layer tapes are released after the layer");
        assert!(compiled.plan.windows.iter().any(|w| w.tape_bytes > 0));
    }

    #[test]
    fn parameter_homes_split_when_no_device_holds_the_buffer() {
        let cfg = MixtralConfig::demo();
        let params = crate::mixtral::parameter_count(&cfg);
        let bytes = params * 4;
        let cluster = Cluster::devices(vec![
            gpu("gpu0", bytes / 2 + (32 << 10), 1 << 20),
            gpu("gpu1", bytes / 2 + (32 << 10), 1 << 20),
        ]);
        let compiled = compile_mixtral(&cfg, 1, 1, &cluster).expect("split homes");
        compiled.plan.check(&cluster, params).unwrap();
        let mut used = [false; 2];
        for home in compiled
            .plan
            .homes
            .iter()
            .filter(|h| h.kind == HomeKind::Param)
        {
            used[home.device] = true;
        }
        assert!(used[0] && used[1]);
    }

    #[test]
    fn a_cluster_smaller_than_the_parameters_is_rejected() {
        let cfg = MixtralConfig::demo();
        let cluster = Cluster::devices(vec![gpu("gpu0", 1024, 1 << 20)]);
        let err = compile_mixtral(&cfg, 1, 1, &cluster).unwrap_err();
        assert!(matches!(err, StageError::ParamsExceedCluster { .. }));
    }

    #[test]
    fn mlp_tape_and_parameters_share_the_cluster() {
        let prog = Program::mlp(2, &[3, 4, 2]);
        let cluster = Cluster::devices(vec![
            cpu("cpu0", 1 << 20, CPU_RUNTIME_BYTES),
            gpu("gpu0", 1 << 20, 1 << 20),
        ]);
        let compiled = compile_program(&prog, &cluster).expect("mlp stages");
        compiled
            .plan
            .check(&cluster, prog.param_len as u64)
            .unwrap();
        assert!(compiled
            .plan
            .homes
            .iter()
            .any(|h| h.kind == HomeKind::Tape && h.bytes() > 0));
        assert!(compiled.gpu_kernels.contains(&Kernel::GemmFwd));
    }

    #[test]
    fn mixtral_backward_follows_the_residual_adjoint() {
        let cfg = MixtralConfig::demo();
        let cluster = Cluster::devices(vec![gpu("gpu0", 64 << 20, 1 << 20)]);
        let compiled = compile_mixtral(&cfg, 2, 8, &cluster).expect("demo");
        for window in &compiled.plan.windows {
            if window.name.contains("ffn-norm") {
                assert_eq!(window.expert, 1, "{}", window.name);
            } else if window.name.contains("attn-norm") {
                assert_eq!(window.expert, 0, "{}", window.name);
            } else if window.name.starts_with("final-norm") {
                assert_eq!(window.expert, 2, "{}", window.name);
            }
        }
        let mut stems = Vec::new();
        for window in compiled.plan.windows.iter().filter(|w| w.pass == Pass::Bwd) {
            let stem = window.name.split('[').next().unwrap().to_string();
            if stems.last() != Some(&stem) {
                stems.push(stem);
            }
        }
        let mut expect = vec![
            "loss".to_string(),
            "lm-head".to_string(),
            "final-norm".to_string(),
        ];
        for layer in (0..cfg.layers).rev() {
            expect.push(format!("L{layer}-add-moe"));
            expect.push(format!("L{layer}-mix"));
            for expert in (0..cfg.experts).rev() {
                expect.push(format!("L{layer}-down-{expert}"));
                expect.push(format!("L{layer}-silu-{expert}"));
                expect.push(format!("L{layer}-up-{expert}"));
                expect.push(format!("L{layer}-gate-{expert}"));
            }
            expect.push(format!("L{layer}-router-gate"));
            expect.push(format!("L{layer}-router"));
            expect.push(format!("L{layer}-ffn-norm"));
            expect.push(format!("L{layer}-add-attn"));
            expect.push(format!("L{layer}-o"));
            expect.push(format!("L{layer}-gqa"));
            expect.push(format!("L{layer}-rope-k"));
            expect.push(format!("L{layer}-rope-q"));
            expect.push(format!("L{layer}-v"));
            expect.push(format!("L{layer}-k"));
            expect.push(format!("L{layer}-q"));
            expect.push(format!("L{layer}-attn-norm"));
        }
        expect.push("embed".to_string());
        assert_eq!(stems, expect);
        let sgd: Vec<_> = compiled
            .plan
            .windows
            .iter()
            .filter(|w| w.exec == Exec::Sgd)
            .collect();
        assert!(sgd.len() > 1, "each parameter home is its own update");
        assert!(sgd.iter().all(|w| w.pass == Pass::Fwd));
        assert!(sgd.iter().all(|w| w.inbound.is_empty() && w.outbound.is_empty()));
    }

    fn linked(a: usize, b: usize, links: &[Link]) -> bool {
        links.iter().any(|link| {
            (link.a == a && link.b == b) || (link.a == b && link.b == a)
        })
    }

    fn hops_follow_links(hops: &[Hop], links: &[Link]) {
        for hop in hops {
            assert!(
                linked(hop.from, hop.to, links),
                "hop {} -> {} is not a link",
                hop.from,
                hop.to
            );
        }
    }

    #[test]
    fn derived_topology_connects_a_cpu_fabric_and_its_gpus() {
        let devices = vec![
            cpu("cpu0", 1 << 20, CPU_RUNTIME_BYTES),
            cpu("cpu1", 1 << 20, CPU_RUNTIME_BYTES),
            gpu("gpu0", 1 << 20, 1 << 20),
            gpu("gpu1", 1 << 20, 1 << 20),
            gpu("gpu2", 1 << 20, 1 << 20),
        ];
        let links = derive_links(&devices);
        assert!(linked(0, 1, &links), "cpus share the fabric");
        assert!(linked(2, 0, &links), "first gpu attaches to cpu0");
        assert!(linked(3, 1, &links), "second gpu attaches to cpu1");
        assert!(linked(4, 0, &links), "third gpu attaches to cpu0");
        assert!(linked(2, 4, &links), "gpus on one cpu can reach each other");
        assert!(!linked(2, 3, &links), "gpus on different cpus meet through the fabric");
        let path = route_message(&links, &[1 << 20, 1 << 20, 0, 0, 0], 2, 3, 64).expect("route");
        assert_eq!(path, vec![2, 0, 1, 3]);
    }

    #[test]
    fn a_bridge_smaller_than_the_message_is_not_a_route() {
        let links = vec![Link { a: 0, b: 1 }, Link { a: 1, b: 2 }];
        let err = route_message(&links, &[0, 16, 0], 0, 2, 32).unwrap_err();
        assert!(matches!(err, StageError::RouteUnavailable { bytes: 32, .. }));
        let path = route_message(&links, &[0, 32, 0], 0, 2, 32).expect("fits");
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn pipeline_messages_cover_forward_adjoint_and_local_updates() {
        let cfg = MixtralConfig::demo();
        let cluster = Cluster::devices(vec![
            cpu("cpu0", 64 << 20, CPU_RUNTIME_BYTES),
            gpu("gpu0", 16 << 10, 1 << 20),
        ]);
        let compiled = compile_mixtral(&cfg, 2, 8, &cluster).expect("demo routes");
        let links = &cluster.links;
        assert!(linked(0, 1, links));
        let mut saw_param = false;
        let mut saw_adjoint = false;
        let mut saw_activation = false;
        for window in &compiled.plan.windows {
            hops_follow_links(&window.inbound, links);
            hops_follow_links(&window.outbound, links);
            if window.pass == Pass::Fwd
                && window.inbound.iter().any(|hop| hop.kind == HopKind::Param)
            {
                saw_param = true;
                assert!(window
                    .inbound
                    .iter()
                    .any(|hop| hop.from == window.home && hop.kind == HopKind::Param)
                    || window.inbound.iter().any(|hop| hop.kind == HopKind::Param));
            }
            if window.pass == Pass::Bwd
                && window
                    .outbound
                    .iter()
                    .any(|hop| hop.kind == HopKind::Adjoint && hop.to == window.home)
            {
                saw_adjoint = true;
            }
            if window.device != 0
                && window
                    .inbound
                    .iter()
                    .any(|hop| hop.kind == HopKind::Activation)
                && window
                    .outbound
                    .iter()
                    .any(|hop| hop.kind == HopKind::Activation)
            {
                saw_activation = true;
            }
            if window.exec == Exec::Sgd {
                assert!(window.inbound.is_empty() && window.outbound.is_empty());
                assert_eq!(window.device, window.home);
            }
        }
        assert!(saw_param, "forward sends parameters to the compute device");
        assert!(saw_adjoint, "backward returns the adjoint to the home");
        assert!(saw_activation, "activations move along the fabric");
        let mut order = Vec::new();
        for window in &compiled.plan.windows {
            for hop in &window.inbound {
                order.push(hop.kind);
            }
            order.push(if window.exec == Exec::Sgd {
                HopKind::Param
            } else if window.pass == Pass::Bwd {
                HopKind::Adjoint
            } else {
                HopKind::Activation
            });
            for hop in &window.outbound {
                order.push(hop.kind);
            }
        }
        let update_at = order.iter().rposition(|kind| *kind == HopKind::Param).unwrap();
        let adjoint_at = order.iter().position(|kind| *kind == HopKind::Adjoint).unwrap();
        assert!(adjoint_at < update_at, "updates follow the adjoint");
    }

    #[test]
    fn an_unreachable_compute_device_is_rejected() {
        let cfg = MixtralConfig::demo();
        let params = crate::mixtral::parameter_count(&cfg) * 4;
        let cluster = Cluster::linked(
            vec![
                gpu("store", params + (1 << 20), 0),
                gpu("run", 1 << 20, 1 << 20),
            ],
            vec![],
        );
        let err = compile_mixtral(&cfg, 2, 8, &cluster).unwrap_err();
        assert!(
            matches!(err, StageError::RouteUnavailable { .. }),
            "{err:?}"
        );
        let linked = Cluster::linked(
            vec![
                gpu("store", params + (1 << 20), 0),
                gpu("run", 1 << 20, 1 << 20),
            ],
            vec![Link { a: 0, b: 1 }],
        );
        let compiled = compile_mixtral(&cfg, 2, 8, &linked).expect("one link is enough");
        assert!(compiled.plan.windows.iter().any(|window| {
            window.home != window.device
                && window
                    .inbound
                    .iter()
                    .any(|hop| hop.kind == HopKind::Param)
        }));
    }
}
