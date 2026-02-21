// Import the shared macro
use crate::include_models;
include_models!(einsum);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::{backend::TestBackend, transpose};

    #[test]
    fn einsum() {
        let model: einsum::Model<TestBackend> = einsum::Model::default();

        let device = Default::default();
        // Run the model with ones as input for easier testing
        let scalar_in = Tensor::<TestBackend, 0>::from_scalar(5.0, &device);
        let inner_prod_in =
            Tensor::<TestBackend, 1>::random([5], Distribution::Normal(0.0, 1.0), &device);
        let input_2d =
            Tensor::<TestBackend, 2>::random([3, 4], Distribution::Normal(0.0, 1.0), &device);
        let batch_diagonal_in =
            Tensor::<TestBackend, 3>::random([3, 5, 5], Distribution::Normal(0.0, 1.0), &device);
        let batch_matmul_in1 =
            Tensor::<TestBackend, 3>::random([5, 2, 3], Distribution::Normal(0.0, 1.0), &device);
        let batch_matmul_in2 =
            Tensor::<TestBackend, 3>::random([5, 3, 4], Distribution::Normal(0.0, 1.0), &device);

        let expected_transpose_out_shape = Shape::from([4, 3]);
        let expected_sum_out_shape = Shape::from([3]);
        let expected_diag_out_shape = Shape::from([3, 5]);
        let expected_inner_prod_out_shape = Shape::from([]);
        let expected_batch_matmul_out_shape = Shape::from([5, 2, 4]);
        let expected_scalar_out_shape = Shape::from([]);

        let (
            output_scalar,
            batch_matmul_out,
            inner_prod_out,
            batch_diagonal_out,
            sum_out,
            transpose_out,
        ) = model.forward(
            scalar_in,
            batch_matmul_in1,
            batch_matmul_in2,
            inner_prod_in.clone(),
            inner_prod_in,
            batch_diagonal_in,
            input_2d.clone(),
            input_2d,
        );

        assert_eq!(transpose_out.shape(), expected_transpose_out_shape);
        assert_eq!(sum_out.shape(), expected_sum_out_shape);
        assert_eq!(batch_diagonal_out.shape(), expected_diag_out_shape);
        assert_eq!(output_inner_prod.shape(), expected_inner_prod_out_shape);
        assert_eq!(output_scalar.shape(), expected_scalar_out_shape);
        assert_eq!(output_batch_matmul.shape(), expected_batch_matmul_out_shape);
    }
}
