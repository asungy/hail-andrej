use std::cell::Cell;
use std::rc::Rc;

#[derive(Debug, Clone)]
enum Op {
    Add(ValueRef, ValueRef),
    Div(ValueRef, ValueRef),
    Exp(ValueRef),
    Mul(ValueRef, ValueRef),
    Pow(ValueRef, ValueRef),
    Sub(ValueRef, ValueRef),
    Tanh(ValueRef),
}

type ValueRef = Rc<Value>;

#[derive(Debug, Clone)]
struct Value {
    pub data: Cell<f64>,
    pub grad: Cell<f64>,
    pub op: Option<Op>,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(data: {data}, gradient: {grad})",
            data = self.data.get(),
            grad = self.grad.get(),
        )
    }
}

impl Value {
    fn new(data: f64) -> ValueRef {
        Rc::new(Value {
            data: Cell::new(data),
            grad: Cell::new(0.),
            op: None,
        })
    }

    fn add(a: &ValueRef, b: &ValueRef) -> ValueRef {
        let a = a.clone();
        let b = b.clone();
        Rc::new(Value {
            data: Cell::new(a.data.get() + b.data.get()),
            grad: Cell::new(0.),
            op: Some(Op::Add(a, b)),
        })
    }

    fn div(num: &ValueRef, den: &ValueRef) -> ValueRef {
        let num = num.clone();
        let den = den.clone();
        Rc::new(Value {
            data: Cell::new(num.data.get() / den.data.get()),
            grad: Cell::new(0.),
            op: Some(Op::Div(num, den)),
        })
    }

    fn exp(value: &ValueRef) -> ValueRef {
        let value = value.clone();
        Rc::new(Value {
            data: Cell::new(value.data.get().exp()),
            grad: Cell::new(0.),
            op: Some(Op::Exp(value)),
        })
    }

    fn mul(a: &ValueRef, b: &ValueRef) -> ValueRef {
        let lhs = a.clone();
        let rhs = b.clone();
        Rc::new(Value {
            data: Cell::new(lhs.data.get() * rhs.data.get()),
            grad: Cell::new(0.),
            op: Some(Op::Mul(lhs, rhs)),
        })
    }

    fn pow(base: &ValueRef, exp: &ValueRef) -> ValueRef {
        let base = base.clone();
        let exp = exp.clone();
        Rc::new(Value {
            data: Cell::new(base.data.get().powf(exp.data.get())),
            grad: Cell::new(0.),
            op: Some(Op::Pow(base, exp)),
        })
    }

    fn sub(a: &ValueRef, b: &ValueRef) -> ValueRef {
        let a = a.clone();
        let b = b.clone();
        Rc::new(Value {
            data: Cell::new(a.data.get() - b.data.get()),
            grad: Cell::new(0.),
            op: Some(Op::Sub(a, b)),
        })
    }

    fn tanh(value: &ValueRef) -> ValueRef {
        let value = value.clone();
        Rc::new(Value {
            data: Cell::new(value.data.get().tanh()),
            grad: Cell::new(0.),
            op: Some(Op::Tanh(value)),
        })
    }

    fn backward(value: &ValueRef) -> () {
        let mut queue = std::collections::VecDeque::<ValueRef>::new();
        let mut visited = std::collections::HashSet::<*const Value>::new();
        value.grad.set(1.);
        queue.push_back(value.clone());
        while let Some(current) = queue.pop_front() {
            if visited.get(&Rc::as_ptr(&current)).is_some() {
                continue;
            }
            if let Some(op) = &current.op {
                match op {
                    Op::Add(a, b) => {
                        a.grad.set(a.grad.get() + current.grad.get());
                        b.grad.set(b.grad.get() + current.grad.get());
                        queue.push_back(a.clone());
                        queue.push_back(b.clone());
                    }
                    Op::Mul(a, b) => {
                        a.grad.set({
                            let dfda = b.data.get();
                            a.grad.get() + (current.grad.get() * dfda)
                        });
                        b.grad.set({
                            let dfdb = a.data.get();
                            b.grad.get() + (current.grad.get() * dfdb)
                        });
                        queue.push_back(a.clone());
                        queue.push_back(b.clone());
                    }
                    Op::Tanh(value) => {
                        // Effectively for f(x) = tanh(x): df/dx = 1-(tanh(x))^2.
                        value.grad.set({
                            // Note: tanh(x) is already calculated in `current`.
                            let dtanh = 1. - current.data.get().powi(2);
                            value.grad.get() + (current.grad.get() * dtanh)
                        });
                        queue.push_back(value.clone());
                    }
                    Op::Exp(value) => {
                        value.grad.set({
                            let dexp = current.data.get();
                            value.grad.get() + (current.grad.get() * dexp)
                        });
                        queue.push_back(value.clone());
                    }
                    Op::Pow(base, exp) => {
                        // Effectively for f(x, y) = x^y: df/dx = yx^(y-1)
                        base.grad.set({
                            // (y-1)
                            let ym1 = exp.data.get() - 1.;
                            // x^(y-1)
                            let xym1 = base.data.get().powf(ym1);
                            // y(x^(y-1))
                            let yxym1 = exp.data.get() * xym1;
                            base.grad.get() + (current.grad.get() * yxym1)
                        });
                        // Effectively for f(x, y) = x^y: df/dy = (x^y)(ln(x))
                        exp.grad.set({
                            // ln(x)
                            let lnx = base.data.get().ln();
                            // x^y. Note: x^y is already calculated in `current`.
                            let xy = current.data.get();
                            // (x^y)(ln(x))
                            let xylnx = xy * lnx;
                            exp.grad.get() + (current.grad.get() * xylnx)
                        });
                        queue.push_back(base.clone());
                        queue.push_back(exp.clone());
                    }
                    Op::Div(num, den) => {
                        // Effectively for f(x, y) = x / y: df/dx = 1 / y
                        num.grad.set({
                            let dfdnum = 1. / den.data.get();
                            num.grad.get() + (current.grad.get() * dfdnum)
                        });
                        // Effectively for f(x, y) = x / y: df/dy = -x / y^2
                        den.grad.set({
                            let y2 = den.data.get().powi(2);
                            let nx = -1. * num.data.get();
                            let dfdden = nx / y2;
                            den.grad.get() + (current.grad.get() * dfdden)
                        });
                        queue.push_back(num.clone());
                        queue.push_back(den.clone());
                    }
                    Op::Sub(lhs, rhs) => {
                        lhs.grad.set(lhs.grad.get() + current.grad.get());
                        rhs.grad.set(rhs.grad.get() + (current.grad.get() * -1.));
                        queue.push_back(lhs.clone());
                        queue.push_back(rhs.clone());
                    }
                }
            }

            visited.insert(Rc::as_ptr(&current));
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
    fn simple_backward() {
        let a = Value::new(2.);
        let b = Value::new(-3.);
        let c = Value::new(10.);
        let e = Value::mul(&a, &b);
        let d = Value::add(&e, &c);
        let f = Value::new(-2.);
        let l = Value::mul(&d, &f);

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
    fn simple_tanh() {
        let x1 = Value::new(2.);
        let x2 = Value::new(0.);
        let w1 = Value::new(-3.);
        let w2 = Value::new(1.);
        let b = Value::new(6.8813735870195432);
        let x1w1 = Value::mul(&x1, &w1);
        let x2w2 = Value::mul(&x2, &w2);
        let x1w1x2w2 = Value::add(&x1w1, &x2w2);
        let n = Value::add(&x1w1x2w2, &b);
        let o = Value::tanh(&n);

        Value::backward(&o);

        let tolerance = 0.00001;
        float_cmp(n.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x1w1.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x2w2.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x2.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x1.grad.get(), -1.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(w1.grad.get(), 1., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(w2.grad.get(), 0., tolerance).unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn complicated_tanh() {
        let x1 = Value::new(2.);
        let x2 = Value::new(0.);
        let w1 = Value::new(-3.);
        let w2 = Value::new(1.);
        let b = Value::new(6.8813735870195432);
        let x1w1 = Value::mul(&x1, &w1);
        let x2w2 = Value::mul(&x2, &w2);
        let x1w1x2w2 = Value::add(&x1w1, &x2w2);
        let n = Value::add(&x1w1x2w2, &b);
        // tanh = (e^(2x) - 1) / (e^(2x) + 1)
        let o = {
            let twox = Value::mul(&Value::new(2.), &n);
            let e2x = Value::exp(&twox);
            let num = Value::sub(&e2x, &Value::new(1.));
            let den = Value::add(&e2x, &Value::new(1.));
            Value::div(&num, &den)
        };

        let tolerance = 0.00001;
        float_cmp(o.data.get(), 0.7071, tolerance).unwrap_or_else(|err| panic!("{err}"));

        Value::backward(&o);

        // Only need to compare leaf nodes gradients.
        float_cmp(x2.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(x1.grad.get(), -1.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(w1.grad.get(), 1., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(w2.grad.get(), 0., tolerance).unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn exp_test() {
        let x = Value::new(2.);
        let f = Value::exp(&x);
        Value::backward(&f);
        let tolerance = 0.001;
        float_cmp(x.grad.get(), 7.389, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(f.grad.get(), 1., tolerance).unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn doubly_referenced_addition() {
        let a = Value::new(3.);
        let b = Value::add(&a, &a);
        Value::backward(&b);
        assert_eq!(a.grad.get(), 2.);
    }

    #[test]
    fn doubly_referenced_multiplication() {
        let a = Value::new(3.);
        let b = Value::mul(&a, &a);
        Value::backward(&b);
        assert_eq!(a.grad.get(), 6.);
    }

    #[test]
    fn fully_connected_layer() {
        let a = Value::new(-2.);
        let b = Value::new(3.);
        let c = Value::add(&a, &b);
        let d = Value::mul(&a, &b);
        let e = Value::mul(&c, &d);
        Value::backward(&e);

        let tolerance = 0.00001;
        float_cmp(a.grad.get(), -3., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(b.grad.get(), -8., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(c.grad.get(), -6., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(d.grad.get(), 1., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(e.grad.get(), 1., tolerance).unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn pow_test() {
        let a = Value::new(2.);
        let b = Value::new(3.);
        let c = Value::pow(&a, &b);

        let tolerance = 0.00001;
        float_cmp(c.data.get(), 8., tolerance).unwrap_or_else(|err| panic!("{err}"));

        Value::backward(&c);

        float_cmp(a.grad.get(), 12., tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(b.grad.get(), 2.0f64.powf(3.) * 2.0f64.ln(), tolerance)
            .unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn div_test() {
        let a = Value::new(4.);
        let b = Value::new(2.);
        let c = Value::div(&a, &b);

        let tolerance = 0.00001;
        float_cmp(c.data.get(), 2., tolerance).unwrap_or_else(|err| panic!("{err}"));

        Value::backward(&c);

        float_cmp(a.grad.get(), 0.5, tolerance).unwrap_or_else(|err| panic!("{err}"));
        float_cmp(b.grad.get(), -1., tolerance).unwrap_or_else(|err| panic!("{err}"));
    }
}
