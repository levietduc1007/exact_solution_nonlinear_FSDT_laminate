import numpy as np
import matplotlib.pyplot as plt
import math

# Material properties
E2 = 12.1e9
E1 = 25 * E2
G12 = 0.5 * E2
G13 = G12
G23 = 0.2 * E2
v12 = 0.25
v21 = v12 * E2 / E1
ks = 5/6

Q11 = E1 / (1 - v12*v21)
Q22 = E2 / (1 - v12*v21)
Q12 = v12 * Q22
Q66 = G12
Q44 = G23
Q55 = G13

Q = np.array([
    [Q11, Q12, 0, 0, 0],
    [Q12, Q22, 0, 0, 0],
    [0, 0, Q66, 0, 0],
    [0, 0, 0, Q44, 0],
    [0, 0, 0, 0, Q55]
])

def transformed_matrix(angle, opt):
    ang = np.deg2rad(angle)
    m = np.round(np.cos(ang), 10)
    n = np.round(np.sin(ang), 10)
    T = np.zeros((5,5))
    if opt == 2:
        T[0:3, 0:3] = np.array([
            [m**2, n**2, m*n],
            [n**2, m**2, -m*n],
            [-2*m*n, 2*m*n, m**2-n**2]
        ])
        T[3:5, 3:5] = np.array([
            [m, -n],
            [n, m]
        ])
    return T

########################################################################
# PHAN PHI TUYEN HINH HOC (VON KARMAN) - CONG THUC (3.4.4, P134)
#
#   eps_xx0 = du0/dx + 1/2 (dw0/dx)^2
#   eps_yy0 = dv0/dy + 1/2 (dw0/dy)^2
#   gam_xy0 = du0/dy + dv0/dx + (dw0/dx)(dw0/dy)
#
# Voi nghiem Navier don mode (m=n=1) thoa man dieu kien bien SS-1:
#   u0   = U  cos(ax) sin(by)
#   v0   = V  sin(ax) cos(by)
#   w0   = W  sin(ax) sin(by)
#   phix = X  cos(ax) sin(by)
#   phiy = Y  sin(ax) cos(by)
#
# He phuong trinh can bang phi tuyen R(q)=0 (q=[U,V,X,Y,W]) duoc suy dan
# CHINH XAC bang tich phan Galerkin (nguyen ly cong ao) tren toan mien
# tam [0,a]x[0,b] bang sympy (khong dung cong thuc gan dung/tra bang),
# sau do rut gon (CSE) thanh ham so hoc ben duoi. Da kiem chung: phan
# tuyen tinh cua R(q) khop CHINH XAC voi ma tran K tuyen tinh o tren, va
# Jacobian giai tich khop sai phan huu han toi 1e-9.
#
# LUU Y QUAN TRONG: nghiem don-mode nay CHI CHINH XAC cho tai SSL
# (q_mn khac 0 chi voi m=n=1). Voi UDL (nhieu mode), cac so hang phi
# tuyen se ghep cac mode lai voi nhau (vd W_11*W_13 -> xuat hien trong
# phuong trinh mode (1,2) v.v.), doi hoi giai dong thoi toan bo he
# Galerkin da-mode (khong con doc lap tung mode 5x5 nhu tuyen tinh nua).
# Phan UDL ben duoi VAN GIU TUYEN TINH nhu code goc; phan SSL duoc bo
# sung ket qua phi tuyen de doi chieu.
########################################################################

def _residual_and_jacobian(U, V, X, Y, W, a, b, q0, A, B, D, Asxz, Asyz):
    """Nonlinear von Karman FSDT residual R(q) and Jacobian dR/dq for the
    single-mode (m=n=1) Navier-Galerkin solution (auto-generated via sympy
    CSE from the exact symbolic Galerkin integration - see derivation
    notes above). A,B,D are 3x3 numpy laminate stiffness matrices;
    Asxz=As[1,1], Asyz=As[0,0] (matching this file's shear convention)."""
    A00 = A[0,0]; A01 = A[0,1]; A02 = A[0,2]
    A10 = A[1,0]; A11 = A[1,1]; A12 = A[1,2]
    A20 = A[2,0]; A21 = A[2,1]; A22 = A[2,2]
    B00 = B[0,0]; B01 = B[0,1]; B02 = B[0,2]
    B10 = B[1,0]; B11 = B[1,1]; B12 = B[1,2]
    B20 = B[2,0]; B21 = B[2,1]; B22 = B[2,2]
    D00 = D[0,0]; D01 = D[0,1]; D02 = D[0,2]
    D10 = D[1,0]; D11 = D[1,1]; D12 = D[1,2]
    D20 = D[2,0]; D21 = D[2,1]; D22 = D[2,2]
    t0 = W**2
    t1 = 16*t0
    t2 = b**2
    t3 = A00*t2
    t4 = a*(A00*U + B00*X)
    t5 = math.pi*t2
    t6 = 9*t5
    t7 = A01*V
    t8 = A22*V
    t9 = B01*Y
    t10 = B22*Y
    t11 = a**2
    t12 = math.pi*t11
    t13 = 9*b*t12
    t14 = A22*t1
    t15 = A22*U
    t16 = 9*math.pi
    t17 = a*t16
    t18 = B22*X
    t19 = (1/36)/(b*t11)
    t20 = math.pi*t19
    t21 = A11*t11
    t22 = A11*V + B11*Y
    t23 = A10*U
    t24 = B10*X
    t25 = a*t6
    t26 = b*t16
    t27 = (1/36)/(a*t2)
    t28 = math.pi*t27
    t29 = B00*t5
    t30 = math.pi**2
    t31 = 9*t2
    t32 = a*t31
    t33 = t30*t32
    t34 = B22*t1
    t35 = math.pi*W
    t36 = Asxz*b
    t37 = a*b
    t38 = V*t30
    t39 = Y*t30
    t40 = 9*t11
    t41 = b*t40
    t42 = B11*t12
    t43 = t30*t41
    t44 = Asyz*a
    t45 = U*t30
    t46 = X*t30
    t47 = b**4
    t48 = A00*t47
    t49 = math.pi**4
    t50 = W**3*t49
    t51 = 81*t50
    t52 = a**4
    t53 = A11*t52
    t54 = 1024*t4
    t55 = t35*t47
    t56 = 1024*b
    t57 = t22*t56
    t58 = t35*t52
    t59 = t2*(A01 + A10 + 4*A22)
    t60 = b**3
    t61 = math.pi*t36
    t62 = math.pi*t44
    t63 = a**3
    t64 = 288*t63
    t65 = 32*W
    t66 = t12*t60
    t67 = t66*(32*t10 + 9*t61 - 32*t7 + 32*t8 - 32*t9)
    t68 = t5*t63
    t69 = t68*(32*t15 + 32*t18 - 32*t23 - 32*t24 + 9*t62)
    t70 = (1/1152)/(t60*t63)
    t71 = t16*t63
    t72 = (1/4)*t30
    t73 = 32*A01
    t74 = -32*A22*W
    t75 = t16*t60
    t76 = 32*A10
    t77 = 9*t30
    t78 = t63*t77
    t79 = B22*t72
    t80 = Asxz*t31
    t81 = D22*t72
    t82 = 32*B01
    t83 = -32*B22*W
    t84 = t60*t77
    t85 = Asyz*t60
    t86 = 32*B10
    t87 = 1024*a
    t88 = -32*A22
    t89 = math.pi*t47
    t90 = -32*B22
    t91 = math.pi*t52
    t92 = t0*t49
    t93 = 243*t92
    R = np.array([
        t20*(-t1*t3 + t11*(-A01*t1 + t14 + t15*t17 + t17*t18) + t13*(t10 + t7 + t8 + t9) + t4*t6),
        t28*(-t1*t21 + t13*t22 + t2*(-A10*t1 + t10*t26 + t14 + t26*t8) + t25*(t15 + t18 + t23 + t24)),
        t19*(-t1*t29 + t12*(-B01*t1 + B22*U*t17 + D22*X*t17 + t34) + t33*(B00*U + D00*X) + t41*(Asxz*X*t37 + B01*t38 + B22*t38 + D01*t39 + D22*t39 + t35*t36)),
        t27*(-t1*t42 + t32*(Asyz*Y*t37 + B10*t45 + B22*t45 + D10*t46 + D22*t46 + t35*t44) + t43*(B11*V + D11*Y) + t5*(-B10*t1 + B22*V*t26 + D22*Y*t26 + t34)),
        t70*(t40*t50*t59 + t48*t51 + t51*t53 - t54*t55 - t57*t58 + t60*t64*(X*t61 + Y*t62 - q0*t37) + t65*t67 + t65*t69),
    ])
    J = np.array([
        [t20*(A00*t25 + A22*t71), t72*(A01 + A22), t20*(B00*t25 + B22*t71), t72*(B01 + B22), t20*(t11*(-W*t73 - t74) - t3*t65)],
        [t72*(A10 + A22), t28*(A11*t13 + A22*t75), t72*(B10 + B22), t28*(B11*t13 + B22*t75), t28*(t2*(-W*t76 - t74) - t21*t65)],
        [t19*(B00*t33 + B22*t78), B01*t72 + t79, t19*(D00*t33 + D22*t78 + t63*t80), D01*t72 + t81, t19*(t12*t80 + t12*(-W*t82 - t83) - t29*t65)],
        [B10*t72 + t79, t27*(B11*t43 + B22*t84), D10*t72 + t81, t27*(D11*t43 + D22*t84 + t40*t85), t27*(Asyz*t12*t31 - t42*t65 + t5*(-W*t86 - t83))],
        [t70*(32*math.pi*W*t2*t63*(-t76 - t88) - t35*t48*t87), t70*(32*math.pi*W*t11*t60*(-t73 - t88) - t35*t53*t56), t70*(Asxz*t64*t89 - B00*t55*t87 + t65*t68*(-t86 - t90)), t70*(-B11*t56*t58 + t65*t66*(-t82 - t90) + 288*t85*t91), t70*(27*t11*t59*t92 + t48*t93 + t53*t93 - t54*t89 - t57*t91 + 32*t67 + 32*t69)],
    ])
    return R, J


def solve_nonlinear_single_mode(a, b, q0, A, B, D, Asxz, Asyz,
                                 tol=1e-10, max_iter=60):
    """Newton-Raphson solve of the von Karman single-mode (m=n=1) system.
    Initial guess = linear solution (q0 kept small effectively at first
    Newton step), returns q=[U,V,X,Y,W] and a converged flag."""
    alpha = np.pi / a
    beta = np.pi / b
    K = np.zeros((5, 5))
    K[0,0]=A[0,0]*alpha**2+A[2,2]*beta**2
    K[0,1]=(A[0,1]+A[2,2])*alpha*beta
    K[0,2]=B[0,0]*alpha**2+B[2,2]*beta**2
    K[0,3]=(B[0,1]+B[2,2])*alpha*beta
    K[1,0]=K[0,1]
    K[1,1]=A[2,2]*alpha**2+A[1,1]*beta**2
    K[1,2]=(B[0,1]+B[2,2])*alpha*beta
    K[1,3]=B[2,2]*alpha**2+B[1,1]*beta**2
    K[2,0]=K[0,2]; K[2,1]=K[1,2]
    K[2,2]=D[0,0]*alpha**2+D[2,2]*beta**2+Asxz
    K[2,3]=(D[0,1]+D[2,2])*alpha*beta
    K[2,4]=Asxz*alpha
    K[3,0]=K[0,3]; K[3,1]=K[1,3]; K[3,2]=K[2,3]
    K[3,3]=D[2,2]*alpha**2+D[1,1]*beta**2+Asyz
    K[3,4]=Asyz*beta
    K[4,0]=K[0,4]=0; K[4,1]=K[1,4]=0; K[4,2]=K[2,4]; K[4,3]=K[3,4]
    K[4,4]=Asxz*alpha**2+Asyz*beta**2
    F = np.array([0, 0, 0, 0, q0], dtype=float)

    q = np.linalg.solve(K, F)   # linear solution as initial guess

    converged = False
    for _ in range(max_iter):
        R, J = _residual_and_jacobian(*q, a, b, q0, A, B, D, Asxz, Asyz)
        dq = np.linalg.solve(J, -R)
        q = q + dq
        if np.max(np.abs(dq)) < tol * max(1.0, np.max(np.abs(q))):
            converged = True
            break
    return q, converged


def run_nonlinear_SSL_comparison():
    """So sanh vong (linear) Navier vs nghiem phi tuyen von Karman
    (don mode m=n=1) cho tai SSL, cung cach chuan hoa w_bar/sigma_bar
    nhu run_table_7_2_1()."""
    phi_list = [[0, 90, 90, 0], [0, 90, 0]]
    side_ratios = [4, 10, 20, 50, 100]
    q0 = 1e6
    h = 0.004

    for phi in phi_list:
        n_layer = len(phi)
        t = h / n_layer

        Q_bar = []
        for p in phi:
            T2 = transformed_matrix(p, 2)
            Q2 = T2.T @ Q @ T2
            Q_bar.append(Q2)

        z1 = [(i - n_layer/2)*t for i in range(n_layer)]
        z2 = [(i+1 - n_layer/2)*t for i in range(n_layer)]

        A = np.zeros((3,3)); B = np.zeros((3,3)); D = np.zeros((3,3)); As = np.zeros((2,2))
        for i in range(n_layer):
            Q_inplane = Q_bar[i][0:3, 0:3]
            Q_shear = Q_bar[i][3:5, 3:5]
            A += Q_inplane * (z2[i] - z1[i])
            B += Q_inplane * (z2[i]**2 - z1[i]**2)/2
            D += Q_inplane * (z2[i]**3 - z1[i]**3)/3
            As += ks * Q_shear * (z2[i] - z1[i])

        epsilon = 1e-6
        A[np.abs(A) < epsilon] = 0.0
        B[np.abs(B) < epsilon] = 0.0
        D[np.abs(D) < epsilon] = 0.0
        As[np.abs(As) < epsilon] = 0.0
        Asxz, Asyz = As[1,1], As[0,0]   # quy uoc giong K matrix goc

        print(f"\nLaminate: {phi} | Load: SSL | LINEAR vs NONLINEAR (von Karman, single-mode)")
        print(f"{'a/h':<5} | {'w_lin':<10} | {'w_NL':<10} | {'sxx_lin':<10} | {'sxx_NL':<10} | {'syy_lin':<10} | {'syy_NL':<10}")
        print("-" * 90)

        for ratio in side_ratios:
            a = h * ratio
            b = a
            alpha = np.pi / a
            beta = np.pi / b
            x_c, y_c = a/2, b/2

            q, ok = solve_nonlinear_single_mode(a, b, q0, A, B, D, Asxz, Asyz)
            if not ok:
                print(f"{ratio:<5} | Newton-Raphson KHONG hoi tu")
                continue
            U_mn, V_mn, X_mn, Y_mn, W_mn = q

            # --- nghiem tuyen tinh (m=n=1, giong code goc) de doi chieu ---
            K = np.zeros((5,5))
            K[0,0]=A[0,0]*alpha**2+A[2,2]*beta**2
            K[0,1]=(A[0,1]+A[2,2])*alpha*beta
            K[0,2]=B[0,0]*alpha**2+B[2,2]*beta**2
            K[0,3]=(B[0,1]+B[2,2])*alpha*beta
            K[1,0]=K[0,1]
            K[1,1]=A[2,2]*alpha**2+A[1,1]*beta**2
            K[1,2]=(B[0,1]+B[2,2])*alpha*beta
            K[1,3]=B[2,2]*alpha**2+B[1,1]*beta**2
            K[2,0]=K[0,2]; K[2,1]=K[1,2]
            K[2,2]=D[0,0]*alpha**2+D[2,2]*beta**2+Asxz
            K[2,3]=(D[0,1]+D[2,2])*alpha*beta
            K[2,4]=Asxz*alpha
            K[3,0]=K[0,3]; K[3,1]=K[1,3]; K[3,2]=K[2,3]
            K[3,3]=D[2,2]*alpha**2+D[1,1]*beta**2+Asyz
            K[3,4]=Asyz*beta
            K[4,0]=0; K[4,1]=0; K[4,2]=K[2,4]; K[4,3]=K[3,4]
            K[4,4]=Asxz*alpha**2+Asyz*beta**2
            Ul, Vl, Xl, Yl, Wl = np.linalg.solve(K, np.array([0,0,0,0,q0], dtype=float))

            def stresses(U_mn, V_mn, X_mn, Y_mn, W_mn):
                exx_c = -U_mn*alpha*np.sin(alpha*x_c)*np.sin(beta*y_c)
                eyy_c = -V_mn*beta*np.sin(alpha*x_c)*np.sin(beta*y_c)
                kxx_c = -X_mn*alpha*np.sin(alpha*x_c)*np.sin(beta*y_c)
                kyy_c = -Y_mn*beta*np.sin(alpha*x_c)*np.sin(beta*y_c)
                z_xx, L_xx = h/2, n_layer-1
                z_yy, L_yy = h/4, 2
                sxx = Q_bar[L_xx][0,0]*(exx_c+z_xx*kxx_c) + Q_bar[L_xx][0,1]*(eyy_c+z_xx*kyy_c)
                syy = Q_bar[L_yy][1,0]*(exx_c+z_yy*kxx_c) + Q_bar[L_yy][1,1]*(eyy_c+z_yy*kyy_c)
                return sxx, syy

            w_bar_l = Wl * 100 * E2 * h**3 / (q0 * a**4)
            w_bar_nl = W_mn * 100 * E2 * h**3 / (q0 * a**4)
            sxx_l, syy_l = stresses(Ul, Vl, Xl, Yl, Wl)
            sxx_nl, syy_nl = stresses(U_mn, V_mn, X_mn, Y_mn, W_mn)
            sxx_bar_l = sxx_l * h**2 / (q0*a**2); sxx_bar_nl = sxx_nl * h**2 / (q0*a**2)
            syy_bar_l = syy_l * h**2 / (q0*a**2); syy_bar_nl = syy_nl * h**2 / (q0*a**2)

            print(f"{ratio:<5} | {w_bar_l:<10.5f} | {w_bar_nl:<10.5f} | {sxx_bar_l:<10.5f} | {sxx_bar_nl:<10.5f} | {syy_bar_l:<10.5f} | {syy_bar_nl:<10.5f}")


def run_table_7_2_1():
    phi_list = [[0, 90, 90, 0], [0, 90, 0]]
    load_types = ['SSL', 'UDL']
    side_ratios = [4, 10, 20, 50, 100]
    
    q0 = 1e6
    h = 0.004

    for phi in phi_list:
        n_layer = len(phi)
        t = h / n_layer

        Q_bar = []
        for p in phi:
            T2 = transformed_matrix(p, 2)
            Q2 = T2.T @ Q @ T2
            Q_bar.append(Q2)

        z1 = [ (i - n_layer/2)*t for i in range(n_layer) ]
        z2 = [ (i+1 - n_layer/2)*t for i in range(n_layer) ]

        A = np.zeros((3,3))
        B = np.zeros((3,3))
        D = np.zeros((3,3))
        As = np.zeros((2,2))

        for i in range(n_layer):
            Q_inplane = Q_bar[i][0:3, 0:3]
            Q_shear = Q_bar[i][3:5, 3:5]
            A += Q_inplane * (z2[i] - z1[i])
            B += Q_inplane * (z2[i]**2 - z1[i]**2)/2
            D += Q_inplane * (z2[i]**3 - z1[i]**3)/3
            As += ks * Q_shear * (z2[i] - z1[i])
            
        # KHỬ SAI SỐ DẤU PHẨY ĐỘNG CHO CÁC MA TRẬN
        epsilon = 1e-6  # Tăng ngưỡng làm tròn để bù trừ sự khuếch đại sai số từ Q
        A[np.abs(A) < epsilon] = 0.0
        B[np.abs(B) < epsilon] = 0.0
        D[np.abs(D) < epsilon] = 0.0
        As[np.abs(As) < epsilon] = 0.0

        for load_type in load_types:
            print(f"\nLaminate: {phi} | Load: {load_type}")
            print(f"{'a/h':<5} | {'w_bar':<10} | {'sig_xx_bar':<10} | {'sig_yy_bar':<10} | {'sig_xy_bar':<10} | {'sig_yz_bar':<10} | {'sig_xz_bar':<10}")
            print("-" * 88)
            
            for ratio in side_ratios:
                a = h * ratio
                b = a
                
                m_max, n_max = (1, 1) if load_type == 'SSL' else (101, 101)
                
                w_center, s_xx_sum, s_yy_sum, s_xy_sum, s_yz_sum, s_xz_sum = 0, 0, 0, 0, 0, 0
                x_c, y_c = a/2, b/2
                
                for m in range(1, m_max+1, 2):
                    for n in range(1, n_max+1, 2):
                        if load_type == 'SSL':
                            q_mn = q0 if m == 1 and n == 1 else 0
                        else:
                            q_mn = 16 * q0 / (np.pi**2 * m * n)
                            
                        if q_mn == 0: continue
                        
                        alpha = m * np.pi / a
                        beta = n * np.pi / b
                        
                        K = np.zeros((5,5))
                        K[0,0] = A[0,0]*alpha**2 + A[2,2]*beta**2
                        K[0,1] = (A[0,1] + A[2,2]) * alpha * beta
                        K[0,2] = B[0,0]*alpha**2 + B[2,2]*beta**2
                        K[0,3] = (B[0,1] + B[2,2]) * alpha * beta
                        K[0,4] = 0
                        
                        K[1,0] = K[0,1]
                        K[1,1] = A[2,2]*alpha**2 + A[1,1]*beta**2
                        K[1,2] = (B[0,1] + B[2,2]) * alpha * beta
                        K[1,3] = B[2,2]*alpha**2 + B[1,1]*beta**2
                        K[1,4] = 0
                        
                        K[2,0] = K[0,2]
                        K[2,1] = K[1,2]
                        K[2,2] = D[0,0]*alpha**2 + D[2,2]*beta**2 + As[1,1]
                        K[2,3] = (D[0,1] + D[2,2]) * alpha * beta
                        K[2,4] = +As[1,1] * alpha
                        
                        K[3,0] = K[0,3]
                        K[3,1] = K[1,3]
                        K[3,2] = K[2,3]
                        K[3,3] = D[2,2]*alpha**2 + D[1,1]*beta**2 + As[0,0]
                        K[3,4] = +As[0,0] * beta
                        
                        K[4,0] = K[0,4]
                        K[4,1] = K[1,4]
                        K[4,2] = K[2,4]
                        K[4,3] = K[3,4]
                        K[4,4] = As[1,1]*alpha**2 + As[0,0]*beta**2
                        
                        F = np.array([0, 0, 0, 0, q_mn])
                        sol = np.linalg.solve(K, F)
                        
                        U_mn, V_mn, X_mn, Y_mn, W_mn = sol
                        
                        w_center += W_mn * np.sin(alpha * a/2) * np.sin(beta * b/2)
                        
                        exx_c = -U_mn * alpha * np.sin(alpha*x_c) * np.sin(beta*y_c)
                        eyy_c = -V_mn * beta * np.sin(alpha*x_c) * np.sin(beta*y_c)
                        exy_corner = (U_mn * beta + V_mn * alpha) * np.cos(alpha*a) * np.cos(beta*b)
                        
                        kxx_c = -X_mn * alpha * np.sin(alpha*x_c) * np.sin(beta*y_c)
                        kyy_c = -Y_mn * beta * np.sin(alpha*x_c) * np.sin(beta*y_c)
                        kxy_corner = (X_mn * beta + Y_mn * alpha) * np.cos(alpha*a) * np.cos(beta*b)
                        
                        gam_yz_mid = (W_mn * beta + Y_mn) * np.sin(alpha*a/2) * np.cos(beta*0)
                        gam_xz_mid = (W_mn * alpha + X_mn) * np.cos(alpha*0) * np.sin(beta*b/2)
                        
                        z_xx = h/2
                        L_xx = n_layer - 1 
                        s_xx_sum += Q_bar[L_xx][0,0]*(exx_c + z_xx*kxx_c) + Q_bar[L_xx][0,1]*(eyy_c + z_xx*kyy_c)
                        
                        z_yy = h/4
                        L_yy = 2
                        s_yy_sum += Q_bar[L_yy][1,0]*(exx_c + z_yy*kxx_c) + Q_bar[L_yy][1,1]*(eyy_c + z_yy*kyy_c)
                        
                        z_xy = -h/2
                        L_xy = 0 
                        s_xy_sum += Q_bar[L_xy][2,2]*(exy_corner + z_xy*kxy_corner)
                        
                        L_yz = 1
                        s_yz_sum += Q_bar[L_yz][3,3]*gam_yz_mid
                        
                        L_xz = 0
                        s_xz_sum += Q_bar[L_xz][4,4]*gam_xz_mid

                w_bar = w_center * 100 * E2 * h**3 / (q0 * a**4)
                sxx_bar = s_xx_sum * h**2 / (q0 * a**2)
                syy_bar = s_yy_sum * h**2 / (q0 * a**2)
                sxy_bar = s_xy_sum * h**2 / (q0 * a**2)
                syz_bar = s_yz_sum * h / (q0 * a)
                sxz_bar = s_xz_sum * h / (q0 * a)
                
                print(f"{ratio:<5} | {w_bar:<10.5f} | {sxx_bar:<10.5f} | {syy_bar:<10.5f} | {sxy_bar:<10.5f} | {syz_bar:<10.5f} | {sxz_bar:<10.5f}")

###### TEST CASE 1: IN MATRIX Q ##############
def print_Q_for_angles():
    for angle in [0, 90]:
        T = transformed_matrix(angle, 2)
        Q_bar = T.T @ Q @ T

        print(f"\nQ_bar matrix for angle = {angle}°:")

        for row in Q_bar:
            print("".join(
                f"{'0' if np.isclose(x, 0) else f'{x:.6e}':>15}"
                for x in row
            ))
##########################################

###### TEST CASE 2: IN MATRIX T ##############

def print_T_for_angles():
    for angle in [0, 90]:
        T2 = transformed_matrix(angle, 2)

        print(f"\n{'='*60}")
        print(f"Angle = {angle}°")
        
        print("\nT2 =")
        for row in T2:
            print("".join(
                f"{'0' if np.isclose(x, 0) else f'{x:.6e}':>15}"
                for x in row
            ))
##########################################

if __name__ == '__main__':
    print_Q_for_angles()
    print_T_for_angles()
    run_table_7_2_1()
    run_nonlinear_SSL_comparison()

