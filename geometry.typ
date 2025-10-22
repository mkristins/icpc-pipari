= Our Geometry Template

== Point class

```cpp
template<class T>
struct Point{
    T x;
    T y;
    Point operator+(const Point &o) const {
        return {x + o.x, y + o.y};
    }
    Point operator-(const Point &o) const {
        return {x - o.x, y - o.y};
    }
    Point operator*(T w) const {
        return {x * w, y * w};
    }
    Point operator/(T w) const {
        return {x / w, y / w};
    }
    Point perp() const {
        return Point{-y, x}; // rotates +90 degrees
    }
    bool operator<(Point &o){
        if(x == o.x) return y < o.y;
        else return x < o.x;
    }
    T cross(Point a) const {
        return x * a.y - y * a.x;
    }
    T dist2() const {
        return x * x + y * y; 
    }
    double dist() const {
        return sqrt(dist2());
    }
    T operator*(const Point &o) const {
        return x*o.x+y*o.y;
    }
};
```

== Cross Product

#image("cross.png")

In this case $arrow(a) times arrow(b) = a_x dot b_y - a_y dot b_x > 0$

== Circumcenter

```cpp
typedef Point<double> P;

double ccRadius(const P& A, const P& B, const P& C) {
    return (B-A).dist()*(C-B).dist()*(A-C).dist()/
    abs((B-A).cross(C-A))/2;
}
P ccCenter(const P& A, const P& B, const P& C) {
    P b = C-A, c = B-A;
    return A + (b*c.dist2()-c*b.dist2()).perp()/b.cross(c)/2;
}
```

== Line Distance

```cpp
typedef Point<double> P;
double lineDist(const P& a, const P& b, const P& p) {
    return (double)(b-a).cross(p-a)/(b-a).dist();
}
```

== Line Intersection

```cpp
typedef Point<double> P;
pair<int, P> lineInter(P s1, P e1, P s2, P e2) {
    auto d = (e1 - s1).cross(e2 - s2);
    if (d == 0) // i f pa ra l l e l
        return {-(s1.cross(e1, s2) == 0), P(0, 0)};
    auto p = s2.cross(e1, e2), q = s2.cross(e2, s1);
    return {1, (s1 * p + e1 * q) / d};
}
```

== Minimum-Enclosing Circle

```cpp
typedef Point<double> P;

pair<P, double> enclose(vector<P> ps) {
    shuffle(ps.begin(), ps.end(), mt19937(time(0)));
    P o = ps[0];
    double r = 0, EPS = 1 + 1e-8;
    int sz = (int)ps.size();
    for(int i = 0 ; i < sz; i ++ ){
        if((o - ps[i]).dist() > r * EPS){
            o = ps[i], r =0;
            for(int j = 0 ; j < i; j ++ ){
                if((o - ps[j]).dist() > r * EPS){
                    o = (ps[i] + ps[j]) / 2;
                    r = (o - ps[i]).dist();
                    for(int k = 0 ; k < j ; k ++ ){
                        if((o - ps[k]).dist() > r * EPS){
                            o = ccCenter(ps[i], ps[j], ps[k]);
                            r = (o - ps[i]).dist();
                        }
                    }
                }
            }
        }
    }
    return {o, r};
}
```

== Polar-Sort

```cpp
sort(X.begin(), X.end(), [&](Point<int> a, Point<int> b){
    Point<int> origin{0, 0};
    bool ba = a < origin, bb = b < origin;
    if(ba != bb) {return ba < bb;}
    else return a.cross(b) > 0;
});
```