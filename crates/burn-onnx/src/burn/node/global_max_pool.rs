use super::prelude::*;

impl NodeCodegen for onnx_ir::node::global_max_pool::GlobalMaxPoolNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        // Determine field type based on input dimension
        let input = self.inputs.first().unwrap();
        let rank = match &input.ty {
            ArgType::Tensor(t) => t.rank,
            _ => panic!("Expected tensor input for GlobalMaxPool"),
        };

        let name = Ident::new(&self.name, Span::call_site());

        let (field_type, init_tokens) = match rank {
            3 => (
                quote! { MaxPool1d },
                quote! {
                    let #name = MaxPool1dConfig::new(1)
                        .init();
                },
            ),
            4 => (
                quote! { MaxPool2d },
                quote! {
                    let #name = MaxPool2dConfig::new([1, 1])
                        .init();
                },
            ),
            dim => panic!("Unsupported input dim ({dim}) for GlobalMaxPoolNode"),
        };

        Some(Field::new(self.name.clone(), field_type, init_tokens))
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        let input = self.inputs.first().unwrap();
        let rank = input.ty.rank();

        match rank {
            3 => {
                imports.register("burn::nn::pool::MaxPool1d");
                imports.register("burn::nn::pool::MaxPool1dConfig");
            }
            4 => {
                imports.register("burn::nn::pool::MaxPool2d");
                imports.register("burn::nn::pool::MaxPool2dConfig");
            }
            dim => panic!("Unsupported input dim ({dim}) for GlobalMaxPoolNode"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::global_max_pool::{GlobalMaxPoolNode, GlobalMaxPoolNodeBuilder};

    fn create_global_max_pool_node_3d(name: &str) -> GlobalMaxPoolNode {
        GlobalMaxPoolNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .build()
    }

    fn create_global_max_pool_node_4d(name: &str) -> GlobalMaxPoolNode {
        GlobalMaxPoolNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .build()
    }

    #[test]
    fn test_global_max_pool_forward_3d() {
        let node = create_global_max_pool_node_3d("pool1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_global_max_pool_forward_4d() {
        let node = create_global_max_pool_node_4d("pool1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.pool1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_global_max_pool_forward_with_clone_3d() {
        let node = create_global_max_pool_node_3d("pool1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            let output = self.pool1.forward(input.clone());
            output
        }
        ");
    }

    #[test]
    fn test_global_max_pool_forward_with_clone_4d() {
        let node = create_global_max_pool_node_4d("pool1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.pool1.forward(input.clone());
            output
        }
        ");
    }
}
