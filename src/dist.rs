//! Run a [`StagePlan`] as a distributed system.
//!
//! A coordinator is the control plane. Each device is a member that connects
//! to the coordinator over TCP and announces its name, kind, and budgets.
//! After every member has joined, the members open peer connections and the
//! coordinator waits until that fabric is up. It then walks the plan: for
//! each window it delivers the inbound hops, tells the owning member to run
//! the window, then delivers the outbound hops. A hop is a framed tile on the
//! peer connection, not a copy inside the coordinator. Parameter homes stay
//! on the member that owns them. SGD runs on the member the plan names, and
//! the updated rectangle is carried back to the home when that member is a
//! different device.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::cluster::{Exec, Hop, HopKind, Pass, StagePlan, Window};

const HELLO: u8 = 1;
const WELCOME: u8 = 2;
const PEERS_UP: u8 = 3;
const PUT: u8 = 4;
const GET: u8 = 5;
const DATA: u8 = 6;
const COMPUTE: u8 = 7;
const SEND: u8 = 8;
const RECV: u8 = 9;
const READY: u8 = 10;
const DONE: u8 = 11;
const FINISH: u8 = 12;

const TIMEOUT: Duration = Duration::from_secs(60);

/// Failure of membership, a hop, or a step acknowledgement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistError {
    pub message: String,
}

impl std::fmt::Display for DistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DistError {}

fn err(message: impl Into<String>) -> DistError {
    DistError {
        message: message.into(),
    }
}

/// Parameter and adjoint elements held by one member, plus the activation
/// tensor carried by the last activation hop.
#[derive(Clone, Debug, Default)]
pub struct Store {
    params: HashMap<u64, f32>,
    adjoint: HashMap<u64, f32>,
    activation: Vec<f32>,
}

impl Store {
    pub fn param(&self, index: u64) -> f32 {
        self.params.get(&index).copied().unwrap_or(0.0)
    }

    pub fn adjoint(&self, index: u64) -> f32 {
        self.adjoint.get(&index).copied().unwrap_or(0.0)
    }

    pub fn set_param(&mut self, index: u64, value: f32) {
        self.params.insert(index, value);
    }

    pub fn set_adjoint(&mut self, index: u64, value: f32) {
        self.adjoint.insert(index, value);
    }
}

/// Work a member runs when the coordinator assigns it a window.
pub trait Compute: Send + Sync {
    fn on_window(&self, rank: usize, window: &Window, store: &mut Store);
}

impl<F> Compute for F
where
    F: Fn(usize, &Window, &mut Store) + Send + Sync,
{
    fn on_window(&self, rank: usize, window: &Window, store: &mut Store) {
        self(rank, window, store);
    }
}

/// Applies `p -= lr * adjoint` on the rectangle of an SGD window.
#[derive(Clone, Copy, Debug)]
pub struct SgdUpdate {
    pub lr: f32,
}

impl Compute for SgdUpdate {
    fn on_window(&self, _rank: usize, window: &Window, store: &mut Store) {
        if window.exec != Exec::Sgd {
            return;
        }
        for index in rect_indices(window.param_offset, window.row_begin, window.row_end, window.col_begin, window.col_end, window.dout)
        {
            let updated = store.param(index) - self.lr * store.adjoint(index);
            store.set_param(index, updated);
        }
    }
}

/// One recorded control-plane event, in the order the coordinator issued it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    Hop {
        from: usize,
        to: usize,
        kind: HopKind,
        bytes: u64,
    },
    Compute {
        device: usize,
        name: String,
        exec: Exec,
        pass: Pass,
    },
}

/// Buffers gathered from the members that own them, plus the control-plane log.
#[derive(Clone, Debug)]
pub struct RunReport {
    pub params: Vec<f32>,
    pub adjoint: Vec<f32>,
    pub log: Vec<Event>,
    pub messages: usize,
    pub members: usize,
}

/// Admit `cluster.devices` members and run `plan`. `params` is the full
/// parameter vector; each home rectangle is delivered to the member that
/// owns it before the first window.
pub fn run<C: Compute + 'static>(
    cluster: &crate::cluster::Cluster,
    plan: &StagePlan,
    params: &[f32],
    compute: C,
) -> Result<RunReport, DistError> {
    if cluster.devices.is_empty() {
        return Err(err("cluster has no devices"));
    }
    let names: Vec<&str> = cluster.devices.iter().map(|d| d.name.as_str()).collect();
    for i in 0..names.len() {
        if names[..i].contains(&names[i]) {
            return Err(err(format!("duplicate device name {}", names[i])));
        }
    }
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| err(e.to_string()))?;
    listener.set_nonblocking(false).ok();
    let addr = listener.local_addr().map_err(|e| err(e.to_string()))?;
    let devices = cluster.devices.clone();
    let plan_thread = plan.clone();
    let params_thread = params.to_vec();
    let compute = Arc::new(compute);
    let mut joins = Vec::new();
    for device in devices.iter().cloned() {
        let compute = Arc::clone(&compute);
        joins.push(thread::spawn(move || member(addr, device, compute)));
    }
    let report = coordinate(listener, &devices, &plan_thread, &params_thread);
    let mut member_err = None;
    for join in joins {
        match join.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => member_err = Some(e),
            Err(_) => member_err = Some(err("member thread panicked")),
        }
    }
    match (report, member_err) {
        (Ok(report), None) => Ok(report),
        (Err(e), _) => Err(e),
        (Ok(_), Some(e)) => Err(e),
    }
}

fn coordinate(
    listener: TcpListener,
    devices: &[crate::cluster::Device],
    plan: &StagePlan,
    params: &[f32],
) -> Result<RunReport, DistError> {
    let n = devices.len();
    let mut streams = Vec::with_capacity(n);
    let mut ports = vec![0u16; n];
    let mut filled = vec![false; n];
    while filled.iter().any(|seen| !seen) {
        let (mut sock, _) = listener.accept().map_err(|e| err(e.to_string()))?;
        prepare(&mut sock)?;
        let (name, kind, buffer, code, port) = read_hello(&mut sock)?;
        let rank = devices
            .iter()
            .position(|d| d.name == name)
            .ok_or_else(|| err(format!("unknown member {name}")))?;
        if filled[rank] {
            return Err(err(format!("duplicate join from {name}")));
        }
        let device = &devices[rank];
        if kind != device.kind as u8
            || buffer != device.buffer_bytes
            || code != device.code_bytes
        {
            return Err(err(format!("member {name} does not match the cluster")));
        }
        streams.push((rank, sock));
        ports[rank] = port;
        filled[rank] = true;
    }
    streams.sort_by_key(|(rank, _)| *rank);
    let mut control: Vec<TcpStream> = streams.into_iter().map(|(_, sock)| sock).collect();
    for (rank, sock) in control.iter_mut().enumerate() {
        write_welcome(sock, rank, &ports)?;
    }
    for sock in control.iter_mut() {
        let tag = read_tag(sock)?;
        if tag != PEERS_UP {
            return Err(err("member did not bring its peer links up"));
        }
    }
    let rects = rectangles(plan);
    for rect in &rects {
        let tile = pack_slice(params, rect);
        write_put(&mut control[rect.home], rect, &tile)?;
        expect_done(&mut control[rect.home])?;
    }
    let mut log = Vec::new();
    let mut messages = 0usize;
    let mut step_id = 0u32;
    for window in &plan.windows {
        for hop in &window.inbound {
            step_id += 1;
            deliver(&mut control, step_id, hop)?;
            messages += 1;
            log.push(Event::Hop {
                from: hop.from,
                to: hop.to,
                kind: hop.kind,
                bytes: hop.bytes,
            });
        }
        step_id += 1;
        write_compute(&mut control[window.device], step_id, window)?;
        expect_done(&mut control[window.device])?;
        log.push(Event::Compute {
            device: window.device,
            name: window.name.clone(),
            exec: window.exec,
            pass: window.pass,
        });
        for hop in &window.outbound {
            step_id += 1;
            deliver(&mut control, step_id, hop)?;
            messages += 1;
            log.push(Event::Hop {
                from: hop.from,
                to: hop.to,
                kind: hop.kind,
                bytes: hop.bytes,
            });
        }
    }
    let mut gathered_p = vec![0.0f32; params.len()];
    let mut gathered_a = vec![0.0f32; params.len()];
    for rect in &rects {
        let values = write_get(&mut control[rect.home], rect, 0)?;
        scatter(&mut gathered_p, rect, &values);
        let values = write_get(&mut control[rect.home], rect, 1)?;
        scatter(&mut gathered_a, rect, &values);
    }
    for sock in control.iter_mut() {
        write_tag(sock, FINISH)?;
    }
    Ok(RunReport {
        params: gathered_p,
        adjoint: gathered_a,
        log,
        messages,
        members: n,
    })
}

fn deliver(control: &mut [TcpStream], step_id: u32, hop: &Hop) -> Result<(), DistError> {
    if hop.from >= control.len() || hop.to >= control.len() {
        return Err(err("hop names a device outside the cluster"));
    }
    write_recv(&mut control[hop.to], step_id, hop)?;
    let ready = read_tag(&mut control[hop.to])?;
    if ready != READY {
        return Err(err("receiver was not ready for a tile"));
    }
    write_send(&mut control[hop.from], step_id, hop)?;
    expect_done(&mut control[hop.from])?;
    expect_done(&mut control[hop.to])?;
    Ok(())
}

fn member(
    addr: std::net::SocketAddr,
    device: crate::cluster::Device,
    compute: Arc<dyn Compute>,
) -> Result<(), DistError> {
    let peers_listen = TcpListener::bind("127.0.0.1:0").map_err(|e| err(e.to_string()))?;
    let port = peers_listen.local_addr().map_err(|e| err(e.to_string()))?.port();
    let mut control = TcpStream::connect(addr).map_err(|e| err(e.to_string()))?;
    prepare(&mut control)?;
    write_hello(&mut control, &device, port)?;
    let (rank, ports) = read_welcome(&mut control)?;
    let mut peers = connect_peers(rank, &ports, &peers_listen)?;
    write_tag(&mut control, PEERS_UP)?;
    let mut store = Store::default();
    loop {
        let frame = read_frame(&mut control)?;
        if frame.is_empty() {
            return Err(err("empty control frame"));
        }
        match frame[0] {
            PUT => {
                let rect = read_rect(&frame[1..])?;
                let bytes = &frame[1 + rect_wire_len()..];
                store.unpack_params(&rect, bytes)?;
                write_tag(&mut control, DONE)?;
            }
            GET => {
                let which = frame[1];
                let rect = read_rect(&frame[2..])?;
                let bytes = store.pack_rect(&rect, which)?;
                write_data(&mut control, &bytes)?;
            }
            COMPUTE => {
                let (step_window, _) = read_window(&frame[1..])?;
                let _step = step_window.0;
                compute.on_window(rank, &step_window.1, &mut store);
                write_tag(&mut control, DONE)?;
            }
            SEND => {
                let (id, dest, hop) = read_transfer(&frame[1..])?;
                let bytes = store.pack_hop(&hop)?;
                let peer = peers.get_mut(&dest).ok_or_else(|| err("missing peer"))?;
                write_tile(peer, id, &bytes)?;
                write_tag(&mut control, DONE)?;
            }
            RECV => {
                let (id, src, hop) = read_transfer(&frame[1..])?;
                write_tag(&mut control, READY)?;
                let peer = peers.get_mut(&src).ok_or_else(|| err("missing peer"))?;
                let (got_id, bytes) = read_tile(peer)?;
                if got_id != id {
                    return Err(err("tile step id does not match the hop"));
                }
                store.unpack_hop(&hop, &bytes)?;
                write_tag(&mut control, DONE)?;
            }
            FINISH => break,
            tag => return Err(err(format!("unexpected control tag {tag}"))),
        }
    }
    Ok(())
}

fn connect_peers(
    rank: usize,
    ports: &[u16],
    listener: &TcpListener,
) -> Result<HashMap<usize, TcpStream>, DistError> {
    let mut peers = HashMap::new();
    for _ in 0..rank {
        let (mut sock, _) = listener.accept().map_err(|e| err(e.to_string()))?;
        prepare(&mut sock)?;
        let mut id = [0u8; 4];
        sock.read_exact(&mut id).map_err(|e| err(e.to_string()))?;
        let other = u32::from_le_bytes(id) as usize;
        peers.insert(other, sock);
    }
    for other in (rank + 1)..ports.len() {
        let mut sock = TcpStream::connect(("127.0.0.1", ports[other])).map_err(|e| err(e.to_string()))?;
        prepare(&mut sock)?;
        sock.write_all(&(rank as u32).to_le_bytes()).map_err(|e| err(e.to_string()))?;
        peers.insert(other, sock);
    }
    Ok(peers)
}

#[derive(Clone, Debug)]
struct Rect {
    home: usize,
    offset: u64,
    row_begin: u64,
    row_end: u64,
    col_begin: u64,
    col_end: u64,
    stride: u64,
}

fn rectangles(plan: &StagePlan) -> Vec<Rect> {
    let mut out = Vec::new();
    for window in &plan.windows {
        if window.param_bytes == 0 {
            continue;
        }
        let rect = Rect {
            home: window.home,
            offset: window.param_offset,
            row_begin: window.row_begin,
            row_end: window.row_end,
            col_begin: window.col_begin,
            col_end: window.col_end,
            stride: window.dout,
        };
        if out.iter().any(|have: &Rect| same_rect(have, &rect)) {
            continue;
        }
        out.push(rect);
    }
    out
}

fn same_rect(a: &Rect, b: &Rect) -> bool {
    a.home == b.home
        && a.offset == b.offset
        && a.row_begin == b.row_begin
        && a.row_end == b.row_end
        && a.col_begin == b.col_begin
        && a.col_end == b.col_end
        && a.stride == b.stride
}

fn rect_indices(offset: u64, row_begin: u64, row_end: u64, col_begin: u64, col_end: u64, stride: u64) -> Vec<u64> {
    let mut out = Vec::new();
    for row in row_begin..row_end {
        for col in col_begin..col_end {
            out.push(offset + row * stride + col);
        }
    }
    out
}

impl Store {
    fn unpack_params(&mut self, rect: &Rect, bytes: &[u8]) -> Result<(), DistError> {
        let values = floats_from(bytes)?;
        let indices = rect_indices(rect.offset, rect.row_begin, rect.row_end, rect.col_begin, rect.col_end, rect.stride);
        if values.len() != indices.len() {
            return Err(err("parameter tile has the wrong length"));
        }
        for (index, value) in indices.into_iter().zip(values) {
            self.params.insert(index, value);
        }
        Ok(())
    }

    fn pack_rect(&self, rect: &Rect, which: u8) -> Result<Vec<u8>, DistError> {
        let indices = rect_indices(rect.offset, rect.row_begin, rect.row_end, rect.col_begin, rect.col_end, rect.stride);
        let mut bytes = Vec::with_capacity(indices.len() * 4);
        for index in indices {
            let value = if which == 1 { self.adjoint(index) } else { self.param(index) };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }

    fn pack_hop(&self, hop: &Hop) -> Result<Vec<u8>, DistError> {
        if hop.kind == HopKind::Activation {
            let n = hop.elems as usize;
            let mut bytes = Vec::with_capacity(n * 4);
            for i in 0..n {
                let value = self.activation.get(i).copied().unwrap_or(0.0);
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            return Ok(bytes);
        }
        let indices = rect_indices(hop.offset, hop.row_begin, hop.row_end, hop.col_begin, hop.col_end, hop.stride);
        let mut bytes = Vec::with_capacity(indices.len() * 4);
        for index in indices {
            let value = if hop.kind == HopKind::Adjoint {
                self.adjoint(index)
            } else {
                self.param(index)
            };
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }

    fn unpack_hop(&mut self, hop: &Hop, bytes: &[u8]) -> Result<(), DistError> {
        let values = floats_from(bytes)?;
        if hop.kind == HopKind::Activation {
            if values.len() != hop.elems as usize {
                return Err(err("activation tile has the wrong length"));
            }
            self.activation = values;
            return Ok(());
        }
        let indices = rect_indices(hop.offset, hop.row_begin, hop.row_end, hop.col_begin, hop.col_end, hop.stride);
        if values.len() != indices.len() {
            return Err(err("tile has the wrong length"));
        }
        for (index, value) in indices.into_iter().zip(values) {
            if hop.kind == HopKind::Adjoint {
                self.adjoint.insert(index, value);
            } else {
                self.params.insert(index, value);
            }
        }
        Ok(())
    }
}

fn pack_slice(params: &[f32], rect: &Rect) -> Vec<u8> {
    let mut bytes = Vec::new();
    for index in rect_indices(rect.offset, rect.row_begin, rect.row_end, rect.col_begin, rect.col_end, rect.stride) {
        let value = params.get(index as usize).copied().unwrap_or(0.0);
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn scatter(dest: &mut [f32], rect: &Rect, values: &[f32]) {
    for (value, index) in values.iter().zip(rect_indices(
        rect.offset,
        rect.row_begin,
        rect.row_end,
        rect.col_begin,
        rect.col_end,
        rect.stride,
    )) {
        if (index as usize) < dest.len() {
            dest[index as usize] = *value;
        }
    }
}

fn floats_from(bytes: &[u8]) -> Result<Vec<f32>, DistError> {
    if bytes.len() % 4 != 0 {
        return Err(err("tile byte length is not a multiple of 4"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks(4) {
        let mut le = [0u8; 4];
        le.copy_from_slice(chunk);
        out.push(f32::from_le_bytes(le));
    }
    Ok(out)
}

fn prepare(sock: &mut TcpStream) -> Result<(), DistError> {
    sock.set_nodelay(true).ok();
    sock.set_read_timeout(Some(TIMEOUT)).map_err(|e| err(e.to_string()))?;
    sock.set_write_timeout(Some(TIMEOUT)).map_err(|e| err(e.to_string()))?;
    Ok(())
}

fn write_frame(sock: &mut TcpStream, payload: &[u8]) -> Result<(), DistError> {
    let len = u32::try_from(payload.len()).map_err(|_| err("frame is too large"))?;
    sock.write_all(&len.to_le_bytes()).map_err(|e| err(e.to_string()))?;
    sock.write_all(payload).map_err(|e| err(e.to_string()))?;
    Ok(())
}

fn read_frame(sock: &mut TcpStream) -> Result<Vec<u8>, DistError> {
    let mut len_buf = [0u8; 4];
    sock.read_exact(&mut len_buf).map_err(|e| err(e.to_string()))?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    sock.read_exact(&mut payload).map_err(|e| err(e.to_string()))?;
    Ok(payload)
}

fn write_tag(sock: &mut TcpStream, tag: u8) -> Result<(), DistError> {
    write_frame(sock, &[tag])
}

fn read_tag(sock: &mut TcpStream) -> Result<u8, DistError> {
    let frame = read_frame(sock)?;
    frame.first().copied().ok_or_else(|| err("empty tag"))
}

fn expect_done(sock: &mut TcpStream) -> Result<(), DistError> {
    let tag = read_tag(sock)?;
    if tag != DONE {
        return Err(err(format!("expected acknowledgement, got {tag}")));
    }
    Ok(())
}

fn write_hello(sock: &mut TcpStream, device: &crate::cluster::Device, port: u16) -> Result<(), DistError> {
    let mut body = vec![HELLO];
    write_str(&mut body, &device.name);
    body.push(match device.kind {
        crate::cluster::DeviceKind::Cpu => 0,
        crate::cluster::DeviceKind::Gpu => 1,
    });
    body.extend_from_slice(&device.buffer_bytes.to_le_bytes());
    body.extend_from_slice(&device.code_bytes.to_le_bytes());
    body.extend_from_slice(&port.to_le_bytes());
    write_frame(sock, &body)
}

fn read_hello(sock: &mut TcpStream) -> Result<(String, u8, u64, u64, u16), DistError> {
    let frame = read_frame(sock)?;
    if frame.first().copied() != Some(HELLO) {
        return Err(err("expected hello"));
    }
    let mut cursor = Cursor::new(&frame[1..]);
    let name = cursor.string()?;
    let kind = cursor.u8()?;
    let buffer = cursor.u64()?;
    let code = cursor.u64()?;
    let port = cursor.u16()?;
    Ok((name, kind, buffer, code, port))
}

fn write_welcome(sock: &mut TcpStream, rank: usize, ports: &[u16]) -> Result<(), DistError> {
    let mut body = vec![WELCOME];
    body.extend_from_slice(&(rank as u32).to_le_bytes());
    body.extend_from_slice(&(ports.len() as u32).to_le_bytes());
    for port in ports {
        body.extend_from_slice(&port.to_le_bytes());
    }
    write_frame(sock, &body)
}

fn read_welcome(sock: &mut TcpStream) -> Result<(usize, Vec<u16>), DistError> {
    let frame = read_frame(sock)?;
    if frame.first().copied() != Some(WELCOME) {
        return Err(err("expected welcome"));
    }
    let mut cursor = Cursor::new(&frame[1..]);
    let rank = cursor.u32()? as usize;
    let n = cursor.u32()? as usize;
    let mut ports = Vec::with_capacity(n);
    for _ in 0..n {
        ports.push(cursor.u16()?);
    }
    Ok((rank, ports))
}

fn rect_wire_len() -> usize {
    4 + 8 * 6
}

fn write_rect(body: &mut Vec<u8>, rect: &Rect) {
    body.extend_from_slice(&(rect.home as u32).to_le_bytes());
    for value in [rect.offset, rect.row_begin, rect.row_end, rect.col_begin, rect.col_end, rect.stride] {
        body.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_rect(bytes: &[u8]) -> Result<Rect, DistError> {
    let mut cursor = Cursor::new(bytes);
    Ok(Rect {
        home: cursor.u32()? as usize,
        offset: cursor.u64()?,
        row_begin: cursor.u64()?,
        row_end: cursor.u64()?,
        col_begin: cursor.u64()?,
        col_end: cursor.u64()?,
        stride: cursor.u64()?,
    })
}

fn write_put(sock: &mut TcpStream, rect: &Rect, tile: &[u8]) -> Result<(), DistError> {
    let mut body = vec![PUT];
    write_rect(&mut body, rect);
    body.extend_from_slice(tile);
    write_frame(sock, &body)
}

fn write_get(sock: &mut TcpStream, rect: &Rect, which: u8) -> Result<Vec<f32>, DistError> {
    let mut body = vec![GET, which];
    write_rect(&mut body, rect);
    write_frame(sock, &body)?;
    let frame = read_frame(sock)?;
    if frame.first().copied() != Some(DATA) {
        return Err(err("expected a gathered tile"));
    }
    floats_from(&frame[1..])
}

fn write_data(sock: &mut TcpStream, bytes: &[u8]) -> Result<(), DistError> {
    let mut body = vec![DATA];
    body.extend_from_slice(bytes);
    write_frame(sock, &body)
}

fn write_compute(sock: &mut TcpStream, step_id: u32, window: &Window) -> Result<(), DistError> {
    let mut body = vec![COMPUTE];
    body.extend_from_slice(&step_id.to_le_bytes());
    write_window(&mut body, window);
    write_frame(sock, &body)
}

fn write_send(sock: &mut TcpStream, step_id: u32, hop: &Hop) -> Result<(), DistError> {
    let mut body = vec![SEND];
    write_transfer(&mut body, step_id, hop.to, hop);
    write_frame(sock, &body)
}

fn write_recv(sock: &mut TcpStream, step_id: u32, hop: &Hop) -> Result<(), DistError> {
    let mut body = vec![RECV];
    write_transfer(&mut body, step_id, hop.from, hop);
    write_frame(sock, &body)
}

fn write_transfer(body: &mut Vec<u8>, step_id: u32, other: usize, hop: &Hop) {
    body.extend_from_slice(&step_id.to_le_bytes());
    body.extend_from_slice(&(other as u32).to_le_bytes());
    write_hop(body, hop);
}

fn read_transfer(bytes: &[u8]) -> Result<(u32, usize, Hop), DistError> {
    let mut cursor = Cursor::new(bytes);
    let id = cursor.u32()?;
    let other = cursor.u32()? as usize;
    let hop = read_hop(&mut cursor)?;
    Ok((id, other, hop))
}

fn write_tile(sock: &mut TcpStream, step_id: u32, bytes: &[u8]) -> Result<(), DistError> {
    let mut body = Vec::with_capacity(4 + bytes.len());
    body.extend_from_slice(&step_id.to_le_bytes());
    body.extend_from_slice(bytes);
    write_frame(sock, &body)
}

fn read_tile(sock: &mut TcpStream) -> Result<(u32, Vec<u8>), DistError> {
    let frame = read_frame(sock)?;
    if frame.len() < 4 {
        return Err(err("short tile"));
    }
    let mut id = [0u8; 4];
    id.copy_from_slice(&frame[..4]);
    Ok((u32::from_le_bytes(id), frame[4..].to_vec()))
}

fn write_window(body: &mut Vec<u8>, window: &Window) {
    write_str(body, &window.name);
    body.push(window.exec as u8);
    body.push(match window.pass {
        Pass::Fwd => 0,
        Pass::Bwd => 1,
    });
    for value in [window.device, window.home, window.layer as usize, window.expert as usize] {
        body.extend_from_slice(&(value as u32).to_le_bytes());
    }
    for value in [
        window.param_offset,
        window.param_bytes,
        window.tape_bytes,
        window.din,
        window.dout,
        window.batch_begin,
        window.batch_end,
        window.row_begin,
        window.row_end,
        window.col_begin,
        window.col_end,
    ] {
        body.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_window(bytes: &[u8]) -> Result<((u32, Window), usize), DistError> {
    let mut cursor = Cursor::new(bytes);
    let step_id = cursor.u32()?;
    let name = cursor.string()?;
    let exec = exec_from(cursor.u8()?)?;
    let pass = if cursor.u8()? == 0 { Pass::Fwd } else { Pass::Bwd };
    let device = cursor.u32()? as usize;
    let home = cursor.u32()? as usize;
    let layer = cursor.u32()?;
    let expert = cursor.u32()?;
    let param_offset = cursor.u64()?;
    let param_bytes = cursor.u64()?;
    let tape_bytes = cursor.u64()?;
    let din = cursor.u64()?;
    let dout = cursor.u64()?;
    let batch_begin = cursor.u64()?;
    let batch_end = cursor.u64()?;
    let row_begin = cursor.u64()?;
    let row_end = cursor.u64()?;
    let col_begin = cursor.u64()?;
    let col_end = cursor.u64()?;
    let window = Window {
        name,
        device,
        code_phases: Vec::new(),
        resident_bytes: 0,
        tape_bytes,
        param_bytes,
        pass,
        exec,
        param_offset,
        batch_begin,
        batch_end,
        row_begin,
        row_end,
        col_begin,
        col_end,
        din,
        dout,
        home,
        layer,
        expert,
        inbound: Vec::new(),
        outbound: Vec::new(),
    };
    Ok(((step_id, window), cursor.i))
}

fn write_hop(body: &mut Vec<u8>, hop: &Hop) {
    body.extend_from_slice(&(hop.from as u32).to_le_bytes());
    body.extend_from_slice(&(hop.to as u32).to_le_bytes());
    body.push(hop.kind as u8);
    for value in [
        hop.bytes,
        hop.offset,
        hop.elems,
        hop.row_begin,
        hop.row_end,
        hop.col_begin,
        hop.col_end,
        hop.stride,
    ] {
        body.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_hop(cursor: &mut Cursor<'_>) -> Result<Hop, DistError> {
    let from = cursor.u32()? as usize;
    let to = cursor.u32()? as usize;
    let kind = match cursor.u8()? {
        0 => HopKind::Param,
        1 => HopKind::Adjoint,
        2 => HopKind::Activation,
        other => return Err(err(format!("unknown hop kind {other}"))),
    };
    Ok(Hop {
        from,
        to,
        kind,
        bytes: cursor.u64()?,
        offset: cursor.u64()?,
        elems: cursor.u64()?,
        row_begin: cursor.u64()?,
        row_end: cursor.u64()?,
        col_begin: cursor.u64()?,
        col_end: cursor.u64()?,
        stride: cursor.u64()?,
    })
}

fn exec_from(tag: u8) -> Result<Exec, DistError> {
    Ok(match tag {
        1 => Exec::Embed,
        2 => Exec::RmsNorm,
        3 => Exec::Linear,
        4 => Exec::RopeQ,
        5 => Exec::RopeK,
        6 => Exec::Gqa,
        7 => Exec::AddAttn,
        8 => Exec::AddMoe,
        9 => Exec::Router,
        10 => Exec::Silu,
        11 => Exec::Mix,
        12 => Exec::Loss,
        13 => Exec::Sgd,
        other => return Err(err(format!("unknown exec {other}"))),
    })
}

fn write_str(body: &mut Vec<u8>, text: &str) {
    let bytes = text.as_bytes();
    body.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    body.extend_from_slice(bytes);
}

struct Cursor<'a> {
    data: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, i: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DistError> {
        if self.i + n > self.data.len() {
            return Err(err("truncated control frame"));
        }
        let out = &self.data[self.i..self.i + n];
        self.i += n;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, DistError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, DistError> {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(buf))
    }

    fn u32(&mut self) -> Result<u32, DistError> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    fn u64(&mut self) -> Result<u64, DistError> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    fn string(&mut self) -> Result<String, DistError> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| err("member name is not utf-8"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::{
        Cluster, Device, DeviceKind, Exec, Hop, HopKind, Pass, StagePlan, Window, CPU_RUNTIME_BYTES,
    };
    use crate::mixtral::{self, MixtralConfig};

    fn cpu(name: &str, buffer: u64) -> Device {
        Device {
            name: name.into(),
            kind: DeviceKind::Cpu,
            buffer_bytes: buffer,
            code_bytes: CPU_RUNTIME_BYTES,
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

    fn blank_window(name: &str, device: usize, home: usize, exec: Exec, pass: Pass) -> Window {
        Window {
            name: name.into(),
            device,
            code_phases: Vec::new(),
            resident_bytes: 0,
            tape_bytes: 0,
            param_bytes: 4,
            pass,
            exec,
            param_offset: 0,
            batch_begin: 0,
            batch_end: 1,
            row_begin: 0,
            row_end: 1,
            col_begin: 0,
            col_end: 1,
            din: 1,
            dout: 1,
            home,
            layer: 0,
            expert: 0,
            inbound: Vec::new(),
            outbound: Vec::new(),
        }
    }

    fn hop(from: usize, to: usize, kind: HopKind) -> Hop {
        Hop {
            from,
            to,
            bytes: 4,
            kind,
            offset: 0,
            elems: 1,
            row_begin: 0,
            row_end: 1,
            col_begin: 0,
            col_end: 1,
            stride: 1,
        }
    }

    #[test]
    fn a_bridge_forwards_the_tile_and_the_update_returns_home() {
        let cluster = Cluster::linked(
            vec![gpu("store", 1 << 20, 0), cpu("bridge", 1 << 20), gpu("run", 1 << 20, 1 << 20)],
            vec![],
        );
        let mut backward = blank_window("linear", 2, 0, Exec::Linear, Pass::Bwd);
        backward.inbound = vec![hop(0, 1, HopKind::Param), hop(1, 2, HopKind::Param)];
        backward.outbound = vec![hop(2, 1, HopKind::Adjoint), hop(1, 0, HopKind::Adjoint)];
        let update = blank_window("sgd", 0, 0, Exec::Sgd, Pass::Fwd);
        let plan = StagePlan {
            homes: Vec::new(),
            windows: vec![backward, update],
            transfers: Vec::new(),
            peak_bytes: vec![0, 0, 0],
        };
        let report = run(&cluster, &plan, &[3.0], |rank: usize, window: &Window, store: &mut Store| {
            if window.pass == Pass::Bwd {
                assert_eq!(rank, 2);
                assert_eq!(store.param(0), 3.0);
                store.set_adjoint(0, store.param(0));
            }
            SgdUpdate { lr: 0.5 }.on_window(rank, window, store);
        })
        .expect("bridge run");
        assert_eq!(report.members, 3);
        assert_eq!(report.messages, 4);
        assert!((report.params[0] - 1.5).abs() < 1e-6);
        assert!((report.adjoint[0] - 3.0).abs() < 1e-6);
        let hops: Vec<_> = report
            .log
            .iter()
            .filter_map(|event| match event {
                Event::Hop { from, to, kind, .. } => Some((*from, *to, *kind)),
                _ => None,
            })
            .collect();
        assert_eq!(
            hops,
            vec![
                (0, 1, HopKind::Param),
                (1, 2, HopKind::Param),
                (2, 1, HopKind::Adjoint),
                (1, 0, HopKind::Adjoint),
            ]
        );
    }

    #[test]
    fn the_demo_schedule_is_coordinated_across_its_members() {
        let cfg = MixtralConfig::demo();
        let cluster = Cluster::devices(vec![cpu("cpu0", 64 << 20), gpu("gpu0", 16 << 10, 1 << 20)]);
        let compiled = crate::cluster::compile_mixtral(&cfg, 2, 8, &cluster).expect("stage");
        let mut params = mixtral::init_params(&cfg);
        mixtral::init_router(&cfg, &mut params);
        let expected_messages: usize = compiled
            .plan
            .windows
            .iter()
            .map(|window| window.inbound.len() + window.outbound.len())
            .sum();
        let report = run(
            &cluster,
            &compiled.plan,
            &params,
            |rank: usize, window: &Window, store: &mut Store| {
                assert_eq!(rank, window.device);
                if window.pass == Pass::Bwd && window.param_bytes > 0 {
                    for index in rect_indices(
                        window.param_offset,
                        window.row_begin,
                        window.row_end,
                        window.col_begin,
                        window.col_end,
                        window.dout,
                    ) {
                        store.set_adjoint(index, store.param(index));
                    }
                }
                SgdUpdate { lr: mixtral::DEMO_LR }.on_window(rank, window, store);
            },
        )
        .expect("demo cluster");
        assert_eq!(report.members, 2);
        assert_eq!(report.messages, expected_messages);
        assert!(report.messages > 0);
        let mut seen_bwd = false;
        let mut seen_sgd = false;
        for event in &report.log {
            if let Event::Compute { exec, pass, device, .. } = event {
                if *exec == Exec::Sgd {
                    assert!(seen_bwd);
                    seen_sgd = true;
                } else if *pass == Pass::Bwd {
                    assert!(!seen_sgd);
                    seen_bwd = true;
                } else {
                    assert!(!seen_bwd && !seen_sgd);
                }
                let _ = device;
            }
        }
        assert!(seen_sgd);
        for (got, start) in report.params.iter().zip(&params) {
            let expect = start - mixtral::DEMO_LR * start;
            assert!((got - expect).abs() < 1e-5, "{got} vs {expect}");
        }
    }
}
