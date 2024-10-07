#set page(flipped: true, numbering: "1",
margin: (
  left: 0.5cm,
  right: 0.5cm
))

#set heading(numbering: "1.")
#set align(center)

LU ICPC komanda "Mazmazītiņie Pipariņi"
- Valters Kļaviņš
- Ansis Gustavs Andersons
- Matīss Kristiņš

#columns(3, gutter: 2em)[
= C++

== Optimizations

```cpp
#pragma GCC optimize("Ofast, unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,tune=native")
```

= Algebra

$ sum_(i=1)^n k^2=(n(n+1)(2n+1))/6 $

]


#pagebreak()
= Organization

#table(
  columns: (20fr, 10fr, 10fr, 10fr, 10fr, 10fr, 10fr, 10fr, 10fr, 10fr, 10fr, 10fr, 10fr, 10fr),
  rows: (2cm, 2cm, 2cm, 2cm, 2cm),
  [], [A], [B], [C], [D], [E], [F], [G], [H], [I], [J], [K], [L], [M],
  [Read], [], [], [], [], [], [], [], [], [], [], [], [], [],
  [Attempted], [], [], [], [], [], [], [], [], [], [], [], [], [],
  [Estimate], [], [], [], [], [], [], [], [], [], [], [], [], [],
  [\#], [], [], [], [], [], [], [], [], [], [], [], [], []
)
#pagebreak()
#image("hex.png")