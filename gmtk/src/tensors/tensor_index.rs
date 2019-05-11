use std::fmt;

#[repr(C)]
pub struct AdvancedSlice {
    pub idx: i64,
    pub start: i64,
    pub end: i64,
    pub step: i64,
    pub is_slice: bool,
    pub is_select_all: bool,
}

pub struct Slice {
    pub start: i64,
    pub end: i64,
    pub step: i64,
}

const MAX_END: i64 = 9223372036854775807;

impl Slice {
    pub fn from(start: i64) -> Slice {
        return Slice {
            start,
            end: MAX_END,
            step: 1,
        };
    }

    pub fn from_wstep(start: i64, step: i64) -> Slice {
        return Slice {
            start,
            end: MAX_END,
            step,
        };
    }

    pub fn to(end: i64) -> Slice {
        return Slice {
            start: 0,
            end,
            step: 1,
        };
    }

    pub fn to_wstep(end: i64, step: i64) -> Slice {
        return Slice {
            start: 0,
            end,
            step
        };
    }

    pub fn step(step: i64) -> Slice {
        return Slice {
            start: 0,
            end: MAX_END,
            step
        }
    }

    pub fn between(start: i64, end: i64) -> Slice {
        return Slice {
            start,
            end,
            step: 1,
        };
    }

    pub fn slice(start: i64, end: i64, step: i64) -> Slice {
        return Slice { start, end, step };
    }
}

impl AdvancedSlice {

    pub fn step(step: i64) -> AdvancedSlice {
        return AdvancedSlice {
            idx: 0,
            start: 0,
            end: 0,
            step,
            is_slice: true,
            is_select_all: false,
        }
    }

    pub fn from(start: i64) -> AdvancedSlice {
        return AdvancedSlice {
            idx: 0,
            start,
            end: MAX_END,
            step: 1,
            is_slice: true,
            is_select_all: false,
        };
    }

    pub fn to(end: i64) -> AdvancedSlice {
        return AdvancedSlice {
            idx: 0,
            start: 0,
            end: end,
            step: 1,
            is_slice: true,
            is_select_all: false,
        };
    }

    pub fn between(start: i64, end: i64) -> AdvancedSlice {
        return AdvancedSlice {
            idx: 0,
            start,
            end,
            step: 1,
            is_slice: true,
            is_select_all: false,
        };
    }

    pub fn all() -> AdvancedSlice {
        return AdvancedSlice {
            idx: 0,
            start: 0,
            end: 0,
            step: 0,
            is_slice: false,
            is_select_all: true,
        };
    }

    pub fn at(idx: i64) -> AdvancedSlice {
        return AdvancedSlice {
            idx,
            start: 0,
            end: 0,
            step: 0,
            is_slice: false,
            is_select_all: false,
        };
    }

    pub fn from_wstep(start: i64, step: i64) -> AdvancedSlice {
        return AdvancedSlice {
            idx: 0,
            start,
            end: MAX_END,
            step,
            is_slice: true,
            is_select_all: false,
        };
    }

    pub fn to_wstep(end: i64, step: i64) -> AdvancedSlice {
        return AdvancedSlice {
            idx: 0,
            start: 0,
            end,
            step,
            is_slice: true,
            is_select_all: false,
        };
    }

    pub fn slice(start: i64, end: i64, step: i64) -> AdvancedSlice {
        return AdvancedSlice {
            idx: 0,
            start,
            end,
            step,
            is_slice: true,
            is_select_all: false,
        };
    }
}

impl fmt::Debug for AdvancedSlice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_select_all {
            return write!(f, "(:)");
        }

        if self.is_slice {
            if self.end == MAX_END {
                return write!(f, "({}::{})", self.start, self.step);
            }

            return write!(f, "({}:{}:{})", self.start, self.end, self.step);
        }

        return write!(f, "({})", self.idx);
    }
}

impl fmt::Debug for Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.end == MAX_END {
            return write!(f, "({}::{})", self.start, self.step);
        }

        return write!(f, "({}:{}:{})", self.start, self.end, self.step);
    }
}

pub trait TensorIndex<Idx: Sized> {
    type Output: Sized;
    fn at(&self, index: Idx) -> Self::Output;
}

pub trait TensorAssign<Idx: Sized, Value: Sized> {
    fn assign(&mut self, idx: Idx, val: Value);
}

#[macro_export]
macro_rules! _slice_ {
    ($array:ident @ $v:expr) => { $array.push(AdvancedSlice::at($v)) };
    ($array:ident @ ;) => { $array.push(AdvancedSlice::all()) };
    ($array:ident @ ;;$s:expr) => { $array.push(AdvancedSlice::step($e)) };
    ($array:ident @ $f:expr;) => { $array.push(AdvancedSlice::from($f)) };
    ($array:ident @ $f:expr;;$s:expr) => { $array.push(AdvancedSlice::from_wstep($f, $s)) };
    ($array:ident @ ;$e:expr) => { $array.push(AdvancedSlice::to($e)) };
    ($array:ident @ ;$e:expr;$s:expr) => { $array.push(AdvancedSlice::to_wstep($e, $s)) };
    ($array:ident @ $f:expr;$e:expr) => { $array.push(AdvancedSlice::between($f, $e)) };
    ($array:ident @ $f:expr;$e:expr;$s:expr) => { $array.push(AdvancedSlice::slice($f, $e, $s)) };

    ($array:ident @ $v:expr, $($tail:tt)+) => {
        $array.push(AdvancedSlice::at($v));
        _slice_!($array @ $($tail)+);
    };
    ($array:ident @ ;, $($tail:tt)+) => {
        $array.push(AdvancedSlice::all());
        _slice_!($array @ $($tail)+);
    };
    ($array:ident @ ;;$s:expr, $($tail:tt)+) => {
        $array.push(AdvancedSlice::step($e));
        _slice_!($array @ $($tail)+);
    };
    ($array:ident @ $f:expr;, $($tail:tt)+) => {
        $array.push(AdvancedSlice::from($f));
        _slice_!($array @ $($tail)+);
    };
    ($array:ident @ $f:expr;;$s:expr, $($tail:tt)+) => {
        $array.push(AdvancedSlice::from_wstep($f, $s));
        _slice_!($array @ $($tail)+);
    };
    ($array:ident @ ;$e:expr, $($tail:tt)+) => {
        $array.push(AdvancedSlice::to($e));
        _slice_!($array @ $($tail)+);
    };
    ($array:ident @ ;$e:expr;$s:expr, $($tail:tt)+) => {
        $array.push(AdvancedSlice::to_wstep($e, $s));
        _slice_!($array @ $($tail)+);
    };
    ($array:ident @ $f:expr;$e:expr, $($tail:tt)+) => {
        $array.push(AdvancedSlice::between($f, $e));
        _slice_!($array @ $($tail)+);
    };
    ($array:ident @ $f:expr;$e:expr;$s:expr, $($tail:tt)+) => {
        $array.push(AdvancedSlice::slice($f, $e, $s));
        _slice_!($array @ $($tail)+);
    };
}

#[macro_export]
macro_rules! slice {
    (;;$s:expr) => { Slice::step($e) };
    ($f:expr;) => { Slice::from($f) };
    ($f:expr;;$s:expr) => { Slice::from_wstep($f, $s) };
    (;$e:expr) => { Slice::to($e) };
    (;$e:expr;$s:expr) => { Slice::to_wstep($e, $s) };
    ($f:expr;$e:expr) => { Slice::between($f, $e) };
    ($f:expr;$e:expr;$s:expr) => { Slice::slice($f, $e, $s) };
    ( $($tail:tt)+ ) => {{
        let mut slices = Vec::new();
        _slice_!(slices @ $($tail)*);
        slices
    }};
}
