#set page(flipped: true, numbering: "1",
margin: (
  left: 0.5cm,
  right: 0.5cm,
  top: 0.5cm,
  bottom: 1cm
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

#outline()

= C++

== Optimizations

```cpp
#pragma GCC optimize("Ofast, unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,tune=native")
```

== Hash function
```cpp
static uint64_t splitmix64(uint64_t x)
{x+=0x9e3779b97f4a7c15;x=(x^(x>>30))*0xbf58476d1ce4e5b9;
x=(x^(x>>27))*0x94d049bb133111eb;
return x^(x>>31);}
struct custom_hash {size_t operator()(uint64_t x) const {
    static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();return splitmix64(x+FIXED_RANDOM);}};
const long long mod=998244353;
//1000000007
long long modpow(long long n, long long m){long long res=1;while(m){if(m&1)res=res*n%mod;n=n*n%mod;m>>=1;}return res;}
```

== C++ random

```cpp
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
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

== Random usable primes

```cpp
666240077 964865333 115091077 378347773 568491163 295451837 658540403 856004729 843998543 380557313
```

= Data Structures

== Treap

```cpp
struct Node{
    int value, cnt, pri; Node *left, *right;
    Node(int p) : value(p), cnt(1), pri(gen()),
        left(NULL), right(NULL) {};
};
typedef Node* pnode;
int get(pnode q){if(!q) return 0; return q->cnt;}
void update_cnt(pnode &q){
    if(!q) return; q->cnt=get(q->left)+get(q->right)+1;
}
void merge(pnode &T, pnode lef, pnode rig){
    if(!lef){T=rig;return;} if(!rig){T=lef;return;}
    if(lef->pri>rig->pri){merge(lef->right,lef->right,rig);T=lef;
    }else{merge(rig->left, lef, rig->left); T = rig;}
    update_cnt(T);
}
void split(pnode cur, pnode &lef, pnode &rig, int key){
    if(!cur){lef=rig=NULL;return;} int id=get(cur->left)+1;
    if(id<=key){split(cur->right,cur->right,rig,key-id);lef=cur;}
    else {split(cur->left, lef, cur->left, key); rig = cur;}
    update_cnt(cur);
}
```

= Algoritms

== Kuhn's algorithm

```cpp
// node matching indexed 1-n with 1-m
const int N = ansus;
vector<int> g[N];
int mt[N], ind[N];
bool used[N];
bool kuhn(int u)
{
    if(used[u])
        return 0;
    used[u]=1;
    for(auto v:g[u])
    {
        if(mt[v]==-1||kuhn(mt[v]))
        {
            mt[v]=u;
            ind[u]=v;
            return 1;
        }
    }
    return 0;
}
int main()
{
    for(int i = 0;i<m;i++)
        mt[i]=-1;
    for(int i = 0;i<n;i++)
        ind[i]=-1;
    for(int run = 1;run;)
    {
        run=0;
        for(int i = 0;i<n;i++)
            used[i]=0;
        for(int i = 0;i<n;i++)
            if(ind[i]==-1&&kuhn(i))
                run=1;
    }
    // ind[u] = -1, ja nav matchots, citadi ind[u] = indekss no otras komponentes
}
```

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

=== Minimum-cost Max-Flow

```cpp
struct Edge
{
    int from, to, capacity, cost;
};

vector<vector<int>> adj, cost, capacity;

const int INF = 1e9;

void shortest_paths(int n, int v0, vector<int>& d, vector<int>& p) {
    d.assign(n, INF);
    d[v0] = 0;
    vector<bool> inq(n, false);
    queue<int> q;
    q.push(v0);
    p.assign(n, -1);

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        inq[u] = false;
        for (int v : adj[u]) {
            if (capacity[u][v] > 0 && d[v] > d[u] + cost[u][v]) {
                d[v] = d[u] + cost[u][v];
                p[v] = u;
                if (!inq[v]) {
                    inq[v] = true;
                    q.push(v);
                }
            }
        }
    }
}

int min_cost_flow(int N, vector<Edge> edges, int K, int s, int t) {
    adj.assign(N, vector<int>());
    cost.assign(N, vector<int>(N, 0));
    capacity.assign(N, vector<int>(N, 0));
    for (Edge e : edges) {
        adj[e.from].push_back(e.to);
        adj[e.to].push_back(e.from);
        cost[e.from][e.to] = e.cost;
        cost[e.to][e.from] = -e.cost;
        capacity[e.from][e.to] = e.capacity;
    }

    int flow = 0;
    int cost = 0;
    vector<int> d, p;
    while (flow < K) {
        shortest_paths(N, s, d, p);
        if (d[t] == INF)
            break;

        // find max flow on that path
        int f = K - flow;
        int cur = t;
        while (cur != s) {
            f = min(f, capacity[p[cur]][cur]);
            cur = p[cur];
        }

        // apply flow
        flow += f;
        cost += f * d[t];
        cur = t;
        while (cur != s) {
            capacity[p[cur]][cur] -= f;
            capacity[cur][p[cur]] += f;
            cur = p[cur];
        }
    }

    if (flow < K)
        return -1;
    else
        return cost;
}
```

= Strings

== Manacher's algorithm longest palindromic substring
```cpp
int manacher(string s){
    int n = s.size(); string p = "^#";
    rep(i,0,n) p += string(1, s[i]) + "#";
    p += "$"; n = p.size(); vector<int> lps(n, 0);
    int C=0, R=0, m=0;
    rep(i,1,n-1){
        int mirr = 2*C - i;
        if(i < R) lps[i] = min(R-i, lps[mirr]);
        while(p[i + 1 + lps[i]] == p[i - 1 - lps[i]]) lps[i]++;
        if(i + lps[i] > R){ C = i; R = i + lps[i]; }
        m = max(m, lps[i]);
    }
    return m;
}
```

== Suffix Array

```cpp

const int M = 26;

void count_sort(vector<int> &p, vector<int> &c)
{
    int n = p.size();
    vector<int> pos(M+1);
    for(auto x:c)
        pos[x+1]++;
    for(int i = 1;i<=M;i++)
        pos[i]+=pos[i-1];
    vector<int> p_new(n);
    for(int i = 0;i<n;i++)
        p_new[pos[c[p[i]]]++]=p[i];
    swap(p,p_new);
}
int main()
{
    fio
    //ifstream cin("in.in");
    int n, m;
    cin >> n >> m;
    vector<int> str(n);
    for(auto &x:str)
        cin >> x;
    str.pb(-1);
    n++;
    vector<int> p(n), c(n);
    {
        vector<pair<char,int> > ve(n);
        for(int i = 0;i<n;i++)
            ve[i]={str[i],i};
        sort(ve.begin(),ve.end());
        for(int i = 0;i<n;i++)
            p[i]=ve[i].se;
        for(int i = 1;i<n;i++)
            c[p[i]]=c[p[i-1]]+(ve[i].fi!=ve[i-1].fi);
    }
    for(int k = 0;(1<<k)<n;k++)
    {
        for(int i = 0;i<n;i++)
            p[i]=(p[i]-(1<<k)+n)%n;
        count_sort(p,c);
        vector<int> c_new(n);
        for(int i = 1;i<n;i++)
            c_new[p[i]]=c_new[p[i-1]]+(c[p[i]]!=c[p[i-1]]||c[(p[i]+(1<<k))%n]!=c[(p[i-1]+(1<<k))%n]);
        swap(c,c_new);
    }
    vector<int> lcp(n);
    int k = 0;
    for(int i = 0;i<n-1;i++)
    {
        int j = p[c[i]-1];
        while(str[i+k]==str[j+k])
            k++;
        lcp[c[i]]=k;
        k=max(k-1,0);
    }
    return 0;
}
```
= Geometry

== Point to Line

Line ($A x + B y + C = 0$) and point $(x_0; y_0)$ distance is:

$display(d = (A x_0 + B y_0 + C)/sqrt(A^2 + B^2))$

== Online Convex Hull trick
```cpp
// KTH notebook
struct Line {
	mutable ll k, m, p;
	bool operator<(const Line& o) const { return k < o.k; }
	bool operator<(ll x) const { return p < x; }
};

struct LineContainer : multiset<Line, less<>> {
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	static const ll inf = LLONG_MAX;
	ll div(ll a, ll b) { // floored division
		return a / b - ((a ^ b) < 0 && a % b); }
	bool isect(iterator x, iterator y) {
		if (y == end()) return x->p = inf, 0;
		if (x->k == y->k) x->p = x->m > y->m ? inf : -inf;
		else x->p = div(y->m - x->m, x->k - y->k);
		return x->p >= y->p;
	}
	void add(ll k, ll m) {
		auto z = insert({k, m, 0}), y = z++, x = y;
		while (isect(y, z)) z = erase(z);
		if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
		while ((y = x) != begin() && (--x)->p >= y->p)
			isect(x, erase(y));
	}
	ll query(ll x) {
		assert(!empty());
		auto l = *lower_bound(x);
		return l.k * x + l.m;
	}
};
```

== Maximum points in a circle of radius R
```cpp
typedef pair<double,bool> pdb;

#define START 0
#define END 1

struct PT 
{
	double x, y; 
	PT() {}
	PT(double x, double y) : x(x), y(y) {}
	PT(const PT &p) : x(p.x), y(p.y)    {}
	PT operator + (const PT &p)  const { return PT(x+p.x, y+p.y); }
	PT operator - (const PT &p)  const { return PT(x-p.x, y-p.y); }
	PT operator * (double c)     const { return PT(x*c,   y*c  ); }
	PT operator / (double c)     const { return PT(x/c,   y/c  ); }
};

PT p[505];
double dist[505][505];
int n, m;

void calcDist()
{
	FOR(i,0,n)
	{
		FOR(j,i+1,n)
			dist[i][j]=dist[j][i]=sqrt((p[i].x-p[j].x)*(p[i].x-p[j].x)
				+(p[i].y-p[j].y)*(p[i].y-p[j].y));
	}
}
int intelInside(int point, double radius)
{
	vector<pdb> ranges;
	FOR(j,0,n)
	{
		if(j==point || dist[j][point]>2*radius) continue;
		double a1=atan2(p[point].y-p[j].y,p[point].x-p[j].x);
		double a2=acos(dist[point][j]/(2*radius));
		ranges.pb({a1-a2,START});
		ranges.pb({a1+a2,END});
	}
	sort(ALL(ranges));
	int cnt=1, ret=cnt;
	for(auto it: ranges)
	{
		if(it.second) cnt--;
		else cnt++;
		ret=max(ret,cnt);
	}

	return ret;
}

int go(double r) 
{
	int cnt=0;
	FOR(i,0,n)
	{
		cnt=max(cnt,intelInside(i,r));
	}
	return cnt;
}
```

== Point in polygon

```cpp
int sideOf(const PT &s, const PT &e, const PT &p) 
{
	ll a = cross(e-s,p-s);
	return (a > 0) - (a < 0);
}

bool onSegment(const PT &s, const PT &e, const PT &p) 
{
	PT ds = p-s, de = p-e;
	return cross(ds,de) == 0 && dot(ds,de) <= 0;
}

/* 
Main routine
Description: Determine whether a point t lies inside a given polygon (counter-clockwise order).
The polygon must be such that every point on the circumference is visible from the first point in the vector.
It returns 0 for points outside, 1 for points on the circumference, and 2 for points inside.
*/

int insideHull2(const vector<PT> &H, int L, int R, const PT &p) {
	int len = R - L;
	if (len == 2) {
		int sa = sideOf(H[0], H[L], p);
		int sb = sideOf(H[L], H[L+1], p);
		int sc = sideOf(H[L+1], H[0], p);
		if (sa < 0 || sb < 0 || sc < 0) return 0;
		if (sb==0 || (sa==0 && L == 1) || (sc == 0 && R == (int)H.size()))
			return 1;
		return 2;
	}
	int mid = L + len / 2;
	if (sideOf(H[0], H[mid], p) >= 0)
		return insideHull2(H, mid, R, p);
	return insideHull2(H, L, mid+1, p);
}

int insideHull(const vector<PT> &hull, const PT &p) {
	if ((int)hull.size() < 3) return onSegment(hull[0], hull.back(), p);
	else return insideHull2(hull, 1, (int)hull.size(), p);
}
```

= Numerical

== FFT

```cpp
using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd> & a, bool invert) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}
vector<int> multiply(vector<int> const& a, vector<int> const& b) {
    vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < a.size() + b.size()) 
        n <<= 1;
    fa.resize(n);
    fb.resize(n);
    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; i++)
        fa[i] *= fb[i];
    fft(fa, true);
    vector<int> result(n);
    for (int i = 0; i < n; i++)
        result[i] = round(fa[i].real());
    return result;
}
```

#block(breakable: false)[
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
== Sum of $n^k$ in $O(k^2)$

```cpp
LL mod;
LL S[105][105];
void solve() {
    LL n, k;
    scanf("%lld %lld %lld", &n, &k, &mod);
    S[0][0] = 1 % mod;
    for (int i = 1; i <= k; i++) {
        for (int j = 1; j <= i; j++) {
            if (i == j) S[i][j] = 1 % mod;
            else S[i][j] = (j * S[i - 1][j] + S[i - 1][j - 1]) % mod;
        }
    }
    LL ans = 0;
    for (int i = 0; i <= k; i++) {
        LL fact = 1, z = i + 1;
        for (LL j = n - i + 1; j <= n + 1; j++) {
            LL mul = j;
            if (mul % z == 0) {
                mul /= z;
                z /= z;
            }
            fact = (fact * mul) % mod;
        }
        ans = (ans + S[k][i] * fact) % mod;
    }
    printf("%lld\n", ans);
}
```

== Gauss method
```cpp
const double EPS = 1e-9;
const int INF = 2; // it doesn't actually have to be infinity or a big number

int gauss (vector < vector<double> > a, vector<double> & ans) {
    int n = (int) a.size();
    int m = (int) a[0].size() - 1;

    vector<int> where (m, -1);
    for (int col=0, row=0; col<m && row<n; ++col) {
        int sel = row;
        for (int i=row; i<n; ++i)
            if (abs (a[i][col]) > abs (a[sel][col]))
                sel = i;
        if (abs (a[sel][col]) < EPS)
            continue;
        for (int i=col; i<=m; ++i)
            swap (a[sel][i], a[row][i]);
        where[col] = row;

        for (int i=0; i<n; ++i)
            if (i != row) {
                double c = a[i][col] / a[row][col];
                for (int j=col; j<=m; ++j)
                    a[i][j] -= a[row][j] * c;
            }
        ++row;
    }

    ans.assign (m, 0);
    for (int i=0; i<m; ++i)
        if (where[i] != -1)
            ans[i] = a[where[i]][m] / a[where[i]][i];
    for (int i=0; i<n; ++i) {
        double sum = 0;
        for (int j=0; j<m; ++j)
            sum += ans[j] * a[i][j];
        if (abs (sum - a[i][m]) > EPS)
            return 0;
    }

    for (int i=0; i<m; ++i)
        if (where[i] == -1)
            return INF;
    return 1;
}
```

= General

== Simulated Annealing
```cpp

const ld T = (ld)2000;
const ld alpha = 0.999999;
// (new_score - old_score) / (temperature_final) ~ 10 works well

const ld L = (ld)1e6;
ld small_rand(){
	return ((ld)gen(L))/L;
}

ld P(ld old, ld nw, ld temp){
	if(nw > old)
		return 1.0;
	return exp((nw-old)/temp);
}

{
  auto start = chrono::steady_clock::now();
  ld time_limit = 2000; 
  ld temperature = T;
  ld max_score = -1;

  while(elapsed_time < time_limit){
    auto cur = chrono::steady_clock::now();
    elapsed_time = chrono::duration_cast<chrono::milliseconds>(cur - start).count();
    temperature *= alpha;
    
    // try a neighboring state
    // ....
    // ....

    old_score = score(old_state);  
    new_score = score(new_state);
    if(P(old_score, new_score, temperature) >= small_rand()){
      old_state = new_state;
      old_score = new_score;
    }
    if(old_score > max_score){
      max_score = old_score;
      max_state = old_state;
    }
  }
}
```

]
#pagebreak()

#image("hex.png")
#image("hex.png")
#image("hex.png")