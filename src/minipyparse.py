from __future__ import annotations
from typing import *
import itertools
import functools
import re

S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
T4 = TypeVar('T4')
T5 = TypeVar('T5')
T6 = TypeVar('T6')
T7 = TypeVar('T7')
T8 = TypeVar('T8')
T9 = TypeVar('T9')
T10 = TypeVar('T10')
T11 = TypeVar('T11')
T12 = TypeVar('T12')
T13 = TypeVar('T13')
T14 = TypeVar('T14')
T15 = TypeVar('T15')
T16 = TypeVar('T16')
T17 = TypeVar('T17')
T18 = TypeVar('T18')
T19 = TypeVar('T19')
T20 = TypeVar('T20')
PF = Callable[[Sequence[S],int],Iterable[Tuple[int,T]]]

L = Optional[Tuple[T,'L[T]']]
def flatten(xs:L[T]) -> List[T]:
	return [n for ns in [xs] for _ in itertools.takewhile(
		lambda _: ns, itertools.count()) if ns for n, ns in [ns]]
def flattenT(xs:L[T]) -> Tuple[T,...]:
	return tuple(n for ns in [xs] for _ in itertools.takewhile(
		lambda _: ns, itertools.count()) if ns for n, ns in [ns])

class P(Generic[S,T]):
	f: PF[S,T] = lambda s, n: []
	def __init__(self, f:Optional[PF[S,T]]=None):
		return next(None for self.f in [(lambda s, n: []) if f is None else f])
	def forward(self, f:PF[S,T]) -> P[S,T]:
		return next(self for self.f in [f])
	@staticmethod
	def okay() -> P[S,None]:
		return P(lambda s, n: [(n, None)])
	@staticmethod
	def pure(x:T) -> P[S,T]:
		return P(lambda s, n: [(n, x)])
	def bind(self, f:Callable[[T],P[S,U]]) -> P[S,U]:
		return P(lambda s, n: (r for i, a in self.f(s, n) for r in f(a).f(s, i)))
	def fmap(self, f:Callable[[T],U]) -> P[S,U]:
		return P(lambda s, n: ((i, f(a)) for i, a in self.f(s, n)))
	def replace(self, v:U) -> P[S,U]:
		return P(lambda s, n: ((i, v) for i, _ in self.f(s, n)))
	def then(self, p:P[S,U]) -> P[S,U]:
		return self.bind(lambda _: p)
	def before(self, p:P[S,U]) -> P[S,T]:
		return self.bind(lambda x: p.replace(x))
	def greedy(self) -> P[S,T]:
		return P(lambda s, n: next(([r] for r in self.f(s, n)), []))
	def alter(self, p:P[S,U]) -> P[S,Union[T,U]]:
		return P(lambda s, n: itertools.chain(self.f(s, n), p.f(s, n)))
	def maybe(self, default:U=cast(Any,None)) -> P[S,Union[T,U]]:
		return self.alter(P.pure(default))
	def many(self) -> P[S,list[T]]:
		return self.some().alter(P.pure([]))
	def some(self) -> P[S,list[T]]:
		return cast(P[S,L[T]], P.fix(lambda p: self.bind(lambda x:
			p.alter(P.pure(None)).fmap(lambda ys: (x, ys))))).fmap(flatten)
	def filter(self, p:Callable[[T],bool]) -> P[S,T]:
		return P(lambda s, n: ((i, a) for i, a in self.f(s, n) if p(a)))
	def skip(self, k:int) -> P[S,T]:
		return P(lambda s, n: ((i + k, a) for i, a in self.f(s, n)))
	def lex(self:P[str,T], some:bool=False) -> P[str,T]:
		return self.before([P.many, P.some][some](P.single(str.isspace)))
	def parse(self, s, i=0, just=False):
		return (x[-1] if just else x for x in self.f(s, i))

	@staticmethod
	def pred(f:Callable[[Sequence[S],int],Optional[Tuple[int,T]]]):
		return P(lambda s, n: [(n + i, a) for r in [f(s,n)] if r for i, a in [r]])
	@staticmethod
	def single(f:Union[S,Callable[[S],bool]]) -> P[S,S]:
		return next(P(lambda s, n: [(n + 1, s[n])] if n < len(s) and p(s[n]) else [])
			for p in [f if callable(f) else lambda x: f == x])
	@staticmethod
	def chunk(s:Sequence[S]) -> P[S,tuple[S,...]]:
		return P.seq(*map(lambda c: P.single(lambda x: x == c), s))
	@staticmethod
	def choice(*xs:P[S,T]) -> P[S,T]:
		return functools.reduce(lambda p, x: p.alter(x), xs)
	@staticmethod
	def regex(pat:Union[str,re.Pattern], group:int=0, flags:int=0) -> P[str,str]:
		return next(P(lambda s, n:
			[(m.end(), m.group(group)) for m in [r.search(cast(str, s), n)] if m])
			for r in [re.compile(pat, flags) if isinstance(pat, str) else pat])
	@staticmethod
	def eof() -> P[S,None]:
		return P(lambda s, i: [] if i < len(s) else [(i, None)])
	@staticmethod
	def fix(f:Callable[[P[S,T]], P[S,T]]) -> P[S,T]:
		return next(p for p in [cast('P[S,T]', P())] for q in [f(p)] for p.f in [q.f])

	@overload
	@staticmethod
	def seq(
		x1:P[S,T1],
	) -> P[S,Tuple[T1]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2],
	) -> P[S,Tuple[T1,T2]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3],
	) -> P[S,Tuple[T1,T2,T3]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
	) -> P[S,Tuple[T1,T2,T3,T4]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
	) -> P[S,Tuple[T1,T2,T3,T4,T5]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5], x6:P[S,T6],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
		x5:P[S,T5], x6:P[S,T6], x7:P[S,T7],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
		x5:P[S,T5], x6:P[S,T6], x7:P[S,T7], x8:P[S,T8],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
		x5:P[S,T5], x6:P[S,T6], x7:P[S,T7], x8:P[S,T8],
		x9:P[S,T9], x10:P[S,T10], x11:P[S,T11],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
		x5:P[S,T5], x6:P[S,T6], x7:P[S,T7], x8:P[S,T8],
		x9:P[S,T9], x10:P[S,T10], x11:P[S,T11], x12:P[S,T12],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16],
		x17:P[S,T17],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16],
		x17:P[S,T17],
		x18:P[S,T18],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16], x17:P[S,T17], x18:P[S,T18], x19:P[S,T19],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19]]: ...
	@overload
	@staticmethod
	def seq(
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16], x17:P[S,T17], x18:P[S,T18], x19:P[S,T19], x20:P[S,T20],
	) -> P[S,Tuple[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20]]: ...
	@overload
	@staticmethod
	def seq(*xs:P[S,T]) -> P[S,Tuple[T,...]]: ...
	@staticmethod
	def seq(*xs): # type: ignore
		return functools.reduce(lambda p, x: x.bind(lambda a:
			p.fmap(lambda b: (a, b))), reversed(xs), P.pure(None)).fmap(flattenT)

	@overload
	@staticmethod
	def apply(
		f:Callable[[T1],U],
		x1:P[S,T1],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2],U],
		x1:P[S,T1], x2:P[S,T2],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5], x6:P[S,T6],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
		x5:P[S,T5], x6:P[S,T6], x7:P[S,T7],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
		x5:P[S,T5], x6:P[S,T6], x7:P[S,T7], x8:P[S,T8],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
		x5:P[S,T5], x6:P[S,T6], x7:P[S,T7], x8:P[S,T8],
		x9:P[S,T9], x10:P[S,T10], x11:P[S,T11],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4],
		x5:P[S,T5], x6:P[S,T6], x7:P[S,T7], x8:P[S,T8],
		x9:P[S,T9], x10:P[S,T10], x11:P[S,T11], x12:P[S,T12],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16],
		x17:P[S,T17],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16],
		x17:P[S,T17],
		x18:P[S,T18],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16], x17:P[S,T17], x18:P[S,T18], x19:P[S,T19],
	) -> P[S,U]: ...
	@overload
	@staticmethod
	def apply(
		f:Callable[[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20],U],
		x1:P[S,T1], x2:P[S,T2], x3:P[S,T3], x4:P[S,T4], x5:P[S,T5],
		x6:P[S,T6], x7:P[S,T7], x8:P[S,T8], x9:P[S,T9], x10:P[S,T10],
		x11:P[S,T11], x12:P[S,T12], x13:P[S,T13], x14:P[S,T14], x15:P[S,T15],
		x16:P[S,T16], x17:P[S,T17], x18:P[S,T18], x19:P[S,T19], x20:P[S,T20],
	) -> P[S,U]: ...
	@staticmethod
	def apply(f, *xs): # type: ignore
		return P.seq(*xs).fmap(lambda ns: f(*ns))

	@overload
	def sepby(self, other:P[S,U], *, keep:Literal[True],
		end:bool=False, some:bool=False) -> P[S,List[Union[T,U]]]: ...
	@overload
	def sepby(self, other:P[S,U], *, keep:Literal[False],
		end:bool=False, some:bool=False) -> P[S,List[T]]: ...
	def sepby(self, other:P[S,U], *, keep:bool=False, end:bool=False, some:bool=False):
		return next(p if some else p.alter(P.pure([]))
			for m in [cast(P[S,List[Union[T,U]]], P.apply(
				(lambda b, c: [b, c]) if keep else (lambda b, c: [c]), other, self))]
			for p in [P.apply(lambda x, y, z: list(itertools.chain(cast(Any, [x]),
				itertools.chain.from_iterable(y), z)),
				self, m.many(), other.fmap(lambda x: [x]).maybe([]) if end else P())])

	def __neg__    (self): return self.maybe()
	def __pos__    (self): return self.some()
	def __invert__ (self): return self.many()
	def __pow__    (self, n): return self.replace(n)
	def __or__     (self, n): return self.alter(n)
	def __rshift__ (self, n): return self.then(n)
	def __lshift__ (self, n): return self.before(n)

def minify() -> str:
	import ast
	with open(__file__) as file:
		body = ast.parse(file.read()).body
	class Rewrite(ast.NodeTransformer):
		def visit_Call(self, node:ast.Call):
			if isinstance(node.func, ast.Name) and node.func.id == 'cast':
				assert len(node.args) == 2 and not len(node.keywords)
				return self.generic_visit(node.args[1])
			return self.generic_visit(node)
		def visit_AnnAssign(self, node:ast.AnnAssign):
			assert isinstance(node.target, ast.Name) and node.value is not None
			return self.generic_visit(ast.Assign([node.target], node.value))
		def visit_FunctionDef(self, node:ast.FunctionDef):
			if any(isinstance(expr, ast.Name) and expr.id == 'overload'
				for expr in node.decorator_list): return None
			assert len(node.body) == 1 and isinstance(node.body[0], ast.Return) \
				and node.body[0].value is not None
			return self.generic_visit(ast.Assign([ast.Name(node.name)],
				ast.Lambda(node.args, node.body[0].value)))
		def visit_ClassDef(self, node:ast.ClassDef):
			body: List[ast.Assign] = [x for item in node.body
				for x in [self.visit(item)] if x is not None]
			assert all(isinstance(x, ast.Assign) and len(x.targets) == 1 for x in body)
			return ast.Assign([ast.Name(node.name)], ast.Call(ast.Name('type'),
				[ast.Constant(node.name), ast.Tuple([ast.Name("object")]), ast.Call(ast.Name('dict'),
					[], [ast.keyword(cast(ast.Name, x.targets[0]).id, x.value) for x in body])], []))
		def visit_arg(self, node:ast.arg):
			return ast.arg(node.arg)
	parts: str = '\n'.join(ast.unparse(
		ast.fix_missing_locations(Rewrite().visit(item))) for item in body
		if (isinstance(item, ast.FunctionDef) and item.name == 'flatten')
		or (isinstance(item, ast.ClassDef) and item.name == 'P'))
	return re.sub(r' ?([()\[\]{},:<>+=]) ?', r'\1', parts).replace('self','Z')

if __name__ == '__main__':
	print(minify())
	d: dict[str,Any] = {}
	exec(minify(), d)
	print(d['P'])

__all__ = ('P', 'minify')

