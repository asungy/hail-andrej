use rand::Rng;
use crate::value::{Value, ValueRef};

struct Neuron {
    weights: Vec<ValueRef>,
    bias: ValueRef,
}

impl Neuron {
    fn new(n: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        Neuron {
            weights: (0..n).map(|_| Value::new(rng.gen_range(-1.0..1.0))).collect(),
            bias: Value::new(rng.gen_range(-1.0..1.0)),
        }
    }
}
