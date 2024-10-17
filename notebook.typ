#set page(flipped: true, numbering: "1",
margin: (
  left: 0.5cm,
  right: 0.5cm,
  top: 0.5cm,
  bottom: 0.5cm
))

#set heading(numbering: "1.")
#set align(center)

#set text(
    size: 10pt,
    font: "New Computer Modern"
)

#block(
    spacing: 2em
)[
    LU ICPC komanda "Mazmazītiņie Pipariņi"
    - Valters Kļaviņš
    - Ansis Gustavs Andersons
    - Matīss Kristiņš
]

#set text(
    size: 8pt,
    font: "New Computer Modern"
)

#set align(left)

#columns(3, gutter: 2em)[
= C++

== Optimizations

```cpp
#pragma GCC optimize("Ofast, unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,tune=native")
```

= Algebra

$ sum_(i=1)^n k^2=(n(n+1)(2n+1))/6 $
$ sum_(i=1)^n k^3=(n(n+1)/2)^2 $

= Number Theory

== Extended GCD

```cpp
int gcd(int a, int b, int& x, int& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    int x1, y1;
    int d = gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return d;
}
```

= Algoritms

== Flows

=== Dinitz

```cpp
struct FlowEdge {
    int v, u;
    ll cap, flow = 0;
    FlowEdge(int v, int u, ll cap) : v(v), u(u), cap(cap) {}
};

struct Dinic {
    const long long flow_inf = 1e18;
    vector<FlowEdge> edges;
    vector<vector<int>> adj;
    int n, m = 0;
    int s, t;
    vector<int> level, ptr;
    queue<int> q;
    Dinic(int n, int s, int t) : n(n), s(s), t(t) {
        adj.resize(n);
        level.resize(n);
        ptr.resize(n);
    }
    void add_edge(int v, int u, ll cap) {
        edges.push_back(v, u, cap);
        edges.push_back(u, v, 0);
        adj[v].push_back(m);
        adj[u].push_back(m + 1);
        m += 2;
    }
    bool bfs() {
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            for (int id : adj[v]) {
                if (edges[id].cap - edges[id].flow < 1)
                    continue;
                if (level[edges[id].u] != -1)
                    continue;
                level[edges[id].u] = level[v] + 1;
                q.push(edges[id].u);
            }
        }
        return level[t] != -1;
    }
    ll dfs(int v, ll pushed) {
        if (pushed == 0)
            return 0;
        if (v == t)
            return pushed;
        for (int& cid = ptr[v]; cid < (int)adj[v].size(); cid++) {
            int id = adj[v][cid];
            int u = edges[id].u;
            if (level[v] + 1 != level[u] || edges[id].cap - edges[id].flow < 1)
                continue;
            ll tr = dfs(u, min(pushed, edges[id].cap - edges[id].flow));
            if (tr == 0)
                continue;
            edges[id].flow += tr;
            edges[id ^ 1].flow -= tr;
            return tr;
        }
        return 0;
    }
    ll flow() {
        ll f = 0;
        while (true) {
            fill(level.begin(), level.end(), -1);
            level[s] = 0;
            q.push(s);
            if (!bfs())
                break;
            fill(ptr.begin(), ptr.end(), 0);
            while (ll pushed = dfs(s, flow_inf)) {
                f += pushed;
            }
        }
        return f;
    }
};
```

#block(breakable: false)[
= Numerical

== NTT
```cpp

const ll mod = (119 << 23) + 1, root = 62; // 998244353
typedef vector<ll> vl;

int modpow(int n, int k);

void ntt(vl &a) {
	int n = a.size(), L = 31 - __builtin_clz(n);
	static vl rt(2, 1);
	for (static int k = 2, s = 2; k < n; k *= 2, s++) {
		rt.resize(n);
		ll z[] = {1, modpow(root, mod >> s)};
		for(int i=k;i<2*k;i++) rt[i] = rt[i / 2] * z[i & 1] % mod;
	}
	vl rev(n);
	for(int i = 0 ; i < n; i ++ ) rev[i] = (rev[i / 2] | (i & 1) << L) / 2;
	for(int i = 0 ; i < n; i ++ ) if (i < rev[i]) swap(a[i], a[rev[i]]);
	for (int k = 1; k < n; k *= 2)
		for (int i = 0; i < n; i += 2 * k) for(int j=0;j<k;j++) {
			ll z = rt[j + k] * a[i + j + k] % mod, &ai = a[i + j];
			a[i + j + k] = ai - z + (z > ai ? mod : 0);
			ai += (ai + z >= mod ? z - mod : z);
		}
}
vl conv(const vl &a, const vl &b) {
	if (a.empty() || b.empty()) return {};
	int s = a.size() + b.size() - 1, B = 32 - __builtin_clz(s),
	    n = 1 << B;
	int inv = modpow(n, mod - 2);
	vl L(a), R(b), out(n);
	L.resize(n), R.resize(n);
	ntt(L), ntt(R);
	for(int i = 0 ; i < n;i ++ )
		out[-i & (n - 1)] = (ll)L[i] * R[i] % mod * inv % mod;
	ntt(out);
	return {out.begin(), out.begin() + s};
}
```
]

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