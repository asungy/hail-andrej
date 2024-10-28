use std::rc::Rc;
use std::cell::Cell;

#[derive(Debug, Clone)]
enum Op {
    Add(ValueRef, ValueRef),
    Mul(ValueRef, ValueRef),
    Tanh(ValueRef),
}

type ValueRef = Rc<Value>;

#[derive(Debug, Clone)]
struct Value {
    pub data: Cell<f64>,
    pub grad: Cell<f64>,
    pub op: Option<Op>,
    pub label: String,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({label}: data: {data}, gradient: {grad})",
            label = if self.label.is_empty() { "<no label>".into() } else { self.label.clone() },
            data = self.data.get(),
            grad = self.grad.get(),
        )
    }
}

impl Value {
    fn new(data: f64, label: &str) -> ValueRef {
        Rc::new(Value {
            data: Cell::new(data),
            grad: Cell::new(0.),
            label: String::from(label),
            op: None,

        })
    }

    fn add(lhs: &ValueRef, rhs: &ValueRef, label: &str) -> ValueRef {
        let lhs = lhs.clone();
        let rhs = rhs.clone();
        Rc::new(Value {
            data: Cell::new(lhs.data.get() + rhs.data.get()),
            grad: Cell::new(0.),
            op: Some(Op::Add(lhs, rhs)),
            label: String::from(label),
        })
    }

    fn mul(lhs: &ValueRef, rhs: &ValueRef, label: &str) -> ValueRef {
        let lhs = lhs.clone();
        let rhs = rhs.clone();
        Rc::new(Value {
            data: Cell::new(lhs.data.get() * rhs.data.get()),
            grad: Cell::new(0.),
            op: Some(Op::Mul(lhs, rhs)),
            label: String::from(label),
        })
    }

    fn tanh(value: &ValueRef, label: &str) -> ValueRef {
        let value = value.clone();
        Rc::new(Value {
            data: Cell::new(value.data.get().tanh()),
            grad: Cell::new(0.),
            op: Some(Op::Tanh(value)),
            label: String::from(label),
        })
    }

    fn backward(value: &ValueRef) -> () {
        let mut queue = std::collections::VecDeque::<ValueRef>::new();
        value.grad.set(1.);
        queue.push_back(value.clone());
        while let Some(current) = queue.pop_front() {
            if let Some(op) = &current.op {
                match op {
                    Op::Add(lhs, rhs) => {
                        lhs.grad.set(lhs.grad.get() + current.grad.get());
                        rhs.grad.set(rhs.grad.get() + current.grad.get());
                        queue.push_back(lhs.clone());
                        queue.push_back(rhs.clone());
                    },
                    Op::Mul(lhs, rhs) => {
                        lhs.grad.set(lhs.grad.get() + (current.grad.get() * rhs.data.get()));
                        rhs.grad.set(rhs.grad.get() + (current.grad.get() * lhs.data.get()));
                        queue.push_back(lhs.clone());
                        queue.push_back(rhs.clone());
                    },
                    Op::Tanh(value) => {
                        // Note: The derivative of tanh is 1-(tanh(x))^2.
                        // We already calculated `tanh` on the forward pass, which is the value of
                        // `current`, so we just square it.
                        value.grad.set(current.grad.get() * (1.-current.data.get().powi(2)));
                        queue.push_back(value.clone());
                    },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Value;

    fn float_cmp(left: f64, right: f64, tolerance: f64) -> Result<(), String> {
        if (left - right).abs() < tolerance {
            Ok(())
        } else {
            Err(format!("float assertion `left == right` failed (tolerance={tolerance}).\n   left: {left}\n  right: {right}\n"))
        }
    }

    #[test]
    fn backprop() {
        let a = Value::new(2., "a");
        let b = Value::new(-3., "b");
        let c = Value::new(10., "c");
        let e = Value::mul(&a, &b, "e");
        let d = Value::add(&e, &c, "d");
        let f = Value::new(-2., "f");
        let l = Value::mul(&d, &f, "L");

        assert_eq!(e.data.get(), -6.);
        assert_eq!(d.data.get(), 4.);
        assert_eq!(l.data.get(), -8.);

        Value::backward(&l);

        assert_eq!(l.grad.get(), 1.);
        assert_eq!(d.grad.get(), -2.);
        assert_eq!(f.grad.get(), 4.);
        assert_eq!(c.grad.get(), -2.);
        assert_eq!(e.grad.get(), -2.);
        assert_eq!(a.grad.get(), 6.);
        assert_eq!(b.grad.get(), -4.);
    }

    #[test]
    fn tanh_test() {
        let x1 = Value::new(2., "x1");
        let x2 = Value::new(0., "x2");
        let w1 = Value::new(-3., "w1");
        let w2 = Value::new(1., "w2");
        let b = Value::new(6.8813735870195432, "b");
        let x1w1 = Value::mul(&x1, &w1, "x1w1");
        let x2w2 = Value::mul(&x2, &w2, "x2w2");
        let x1w1x2w2 = Value::add(&x1w1, &x2w2, "x1w1x2w2");
        let n = Value::add(&x1w1x2w2, &b, "n");
        let o = Value::tanh(&n, "o");

        Value::backward(&o);

        let tolerance = 0.00001;
        float_cmp(n.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x1w1.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x2w2.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x2.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x1.grad.get(), -1.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(w1.grad.get(), 1., tolerance).unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn doubly_referenced_addition() {
        let a = Value::new(3., "a");
        let b = Value::add(&a, &a, "b");
        Value::backward(&b);
        assert_eq!(a.grad.get(), 2.);
    }

    #[test]
    fn doubly_referenced_multiplication() {
        let a = Value::new(3., "a");
        let b = Value::mul(&a, &a, "b");
        Value::backward(&b);
        assert_eq!(a.grad.get(), 6.);
    }

    #[test]
    fn fully_connected_layer() {
        let a = Value::new(-2., "a");
        let b = Value::new(3., "b");
        let c = Value::add(&a, &b, "c");
        let d = Value::mul(&a, &b, "d");
        let e = Value::mul(&c, &d, "f");
        Value::backward(&e);

        let tolerance = 0.00001;
        float_cmp(a.grad.get(), -3., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(b.grad.get(), -8., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(c.grad.get(), -6., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(d.grad.get(), 1., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(e.grad.get(), 1., tolerance).unwrap_or_else(|err| panic!("{err}"));
    }
}
