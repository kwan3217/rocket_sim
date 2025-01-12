"""
Foo all the Bars
"""

def poly_result(order=5, d=0, coef=None):
    product = order
    if coef is not None:
        product*=coef
    if d<order:
        return r"\infty"
    elif d==order:
        for coefd in range(order - 1, 0, -1):
            product *= coefd
        return str(product)
    else:
        return "0"

def poly_deriv_tex(order=5,d=0,coef=None,mark_coef=True):
    open_coef=" {{ " if mark_coef else ""
    close_coef=" }} " if mark_coef else ""
    dorder=order-d
    if dorder<0:
        return open_coef+"0"+close_coef
    if d == 0:
        coefs = str(coef) if coef is not None else ""
        primes = ""
    else:
        coefs = open_coef+"%d" % order+close_coef
        for i_d in range(d-1):
            coefd=order-i_d-1
            coefs = open_coef+"%d"%coefd+close_coef+open_coef+r"\cdot"+close_coef + coefs
        if coef is not None:
            coefs=coefs+"\cdot "+str(coef)
    tex = coefs + ((r"{{x}}^{{%d}}" % dorder if dorder > 1 else r"{{x}}") if dorder > 0 else "")
    return tex

def lhpoly_tex(name='f', orders=5, d=0, coef=None, mark_coef=True, mark_poly=False):
    try:
        for order in orders:
            pass
    except TypeError:
        orders=[orders]
    open_poly=" {{ " if mark_poly else ""
    close_poly=" }} " if mark_poly else ""
    primes = "'"*d
    p=""
    maxorder=0
    for order in orders:
        if order>maxorder:
            maxorder=order
        if p!="":
            p+="+"
        p+=poly_deriv_tex(order,d,coef=coef,mark_coef=mark_coef)
    tex = (r"{{" + name + r"}}" + primes + r"{{(x)}} &= " +
           open_poly+p+close_poly + r"\\" +
           r" &= {{ " + poly_result(maxorder,coef=coef,d=d))+" }} "
    return tex


def main():
    for d in range(10):
        print(lhpoly_tex('f',[4,2,1], d))


if __name__ == "__main__":
    main()