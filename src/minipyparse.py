from __future__ import annotations
from typing import *
import itertools
import functools
import re

S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')
PF = Callable[[Sequence[S],int],Iterable[Tuple[int,T]]]

L = Optional[Tuple[T,'L[T]']]
def fromL(xs:L[T]) -> List[T]:
	return [n for ns in [xs] for _ in itertools.takewhile(
		lambda _: ns, itertools.count()) if ns for n, ns in [ns]]

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
	def maybe(self) -> P[S,Optional[T]]:
		return self.alter(P.pure(None))
	def many(self) -> P[S,L[T]]:
		return self.some().maybe()
	def some(self) -> P[S,L[T]]:
		return P.fix(lambda p: self.bind(lambda x: p.alter(P.pure(None)).fmap(lambda ys: (x, ys))))
	@staticmethod
	def seq(*xs:P[S,T]) -> P[S,L[T]]:
		return functools.reduce(lambda p, x:
			x.bind(lambda a: p.fmap(lambda b: (a, b))), reversed(xs), P.pure(None))
	def skip(self, k:int) -> P[S,T]:
		return P(lambda s, n: ((i + k, a) for i, a in self.f(s, n)))
	def lex(self:P[str,T], force:bool=False) -> P[str,T]:
		return self.before([P.many, P.some][force](P.single(str.isspace)))
	@staticmethod
	def pred(f:Callable[[Sequence[S],int],Optional[Tuple[int,T]]]):
		return P(lambda s, n: [(n + i, a) for r in [f(s,n)] if r for i, a in [r]])
	@staticmethod
	def single(f:Union[S,Callable[[S],bool]]) -> P[S,S]:
		return next(P(lambda s, n: [(n + 1, s[n])] if n < len(s) and p(s[n]) else [])
			for p in [f if callable(f) else lambda x: f == x])
	def flat(self:P[S,L[T]]) -> P[S,list[T]]:
		return self.fmap(fromL)
	@staticmethod
	def chunk(s:Sequence[S]) -> P[S,L[S]]:
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
	def parse(self, s, i=0, just=False):
		return (x[-1] if just else x for x in self.f(s, i))
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
	cls = next(item for item in body
		if isinstance(item, ast.ClassDef) and item.name == 'P')
	items: dict[str,ast.expr] = {}
	for item in cls.body:
		if isinstance(item, ast.AnnAssign):
			assert isinstance(item.target, ast.Name)
			assert item.value is not None
			items[item.target.id] = item.value
		elif isinstance(item, ast.FunctionDef):
			assert len(item.body) == 1
			assert isinstance(item.body[0], ast.Return)
			def untype(arg):
				if isinstance(arg, list):
					for x in arg: untype(x)
				elif isinstance(arg, ast.arg) and hasattr(arg, 'annotation'):
					arg.annotation = None
			for k, v in vars(item.args).items(): untype(v)
			item.returns = None
			assert item.body[0].value is not None
			items[item.name] = functools.reduce(
				lambda x, f: ast.Call(f, [x], []),
				item.decorator_list,
				cast(ast.expr, ast.Lambda(item.args, item.body[0].value)))
		else: assert False
	parts: list[str] = []
	for item in body:
		if isinstance(item, ast.FunctionDef) and item.name == 'fromL':
			assert len(item.body) == 1
			assert isinstance(item.body[0], ast.Return)
			assert item.body[0].value is not None
			parts.extend(['fromL=',
				ast.unparse(ast.Lambda(item.args, item.body[0].value))])
	parts.append('\nP=type("P",(object,),{')
	for k, v in items.items():
		parts.extend([repr(k), ':', ast.unparse(v), ','])
	if len(parts) >= 2: parts.pop()
	parts.append('})')
	return re.sub(r' ?([()\[\]{},:<>+=]) ?', r'\1', ''.join(parts)).replace('self','Z')
