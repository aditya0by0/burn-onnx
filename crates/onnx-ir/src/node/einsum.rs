//! # Einsum
//!
//! Applies global average pooling to the input tensor.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Einsum.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

pub enum EinsumEquation {
    /// Example: "ij,jk->ik"
    Subscript(String),
    /// Example: "abc,cd->abd"
    SubscriptWithEllipsis(String),
    /// Example: "abc,cd->abd"
    SubscriptWithEllipsisAndOutput(String),
}

/// Node representation for Einsum operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct EinsumNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub equation: EinsumEquation,
}
