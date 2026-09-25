//! Dependent optics, following Pietro Vertechi, arXiv:2204.09547.
//!
//! The modules follow the paper, then the constructions it cites:
//!
//! - [`finset`], [`span`], [`slice`] — finite sets, spans, and the slice indexed category
//! - [`optic`], [`dlens`], [`dprism`] — Definition 1, dependent lenses (Definition 2), dependent prisms
//! - [`functor_lens`], [`mixed`] — Propositions 2 and 1
//! - [`monoid`], [`comonoid`], [`monoidal_optic`] — Sections 3.2 and 3.3
//! - [`closed`], [`bimod`] — Section 3.4
//! - [`tambara`] — Section 4
//! - [`bicat_optic`], [`fibre`] — Remark 3 and fibre optics
//! - [`enriched`] — Remark 1
//! - [`polynomial`] — Milewski's polynomial optics, as dependent optics
//! - [`monoidal_structure`] — Section 5
//! - [`ad`] — reverse-mode differentiation with an explicit residual (Section 3.1)
//! - [`kernel`] — MLP, ResNet, and transformer blocks compiled to NVIDIA kernels
//! - [`mixtral`] — Mixtral 8x7B (sliding-window GQA, RoPE, RMSNorm, top-2 SwiGLU experts) compiled for SGD
//! - [`cluster`] — stages parameter buffers and tapes onto CPUs and GPUs under code and buffer budgets
//!
//! A compiled model is a schedule: kernel text in code memory, and parameters,
//! adjoints, residuals, and scratch in buffer memory. Each launch names the
//! slices it reads and writes. A cluster assigns those slices to devices, and
//! the Mixtral emitter runs that assignment.
//! - [`vect`] — finite-dimensional vector spaces over `GF(2)`, shared by the linear examples

pub mod ad;
pub mod bicat_optic;
pub mod bimod;
pub mod closed;
pub mod cluster;
pub mod comonoid;
pub mod dlens;
pub mod dprism;
pub mod enriched;
pub mod fibre;
pub mod finset;
pub mod functor_lens;
pub mod kernel;
pub mod mixed;
pub mod mixtral;
pub mod monoid;
pub mod monoidal_optic;
pub mod monoidal_structure;
pub mod optic;
pub mod polynomial;
pub mod slice;
pub mod span;
pub mod tambara;
pub mod vect;

pub use finset::MAX_CARD;
