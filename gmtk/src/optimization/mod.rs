pub mod example;
pub mod batch_example;
pub mod accumulators;
pub mod simple_gradient_descent;
pub mod numerical_gradient;
pub mod optim_traits;
pub mod adam;
pub mod sgd;
pub mod early_stopping;

pub use self::example::*;
pub use self::batch_example::*;
pub use self::accumulators::*;
pub use self::simple_gradient_descent::*;
pub use self::optim_traits::*;
pub use self::adam::Adam;
pub use self::sgd::SGD;
pub use self::early_stopping::*;