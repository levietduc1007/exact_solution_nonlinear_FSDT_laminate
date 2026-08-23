# Exact Solution for Nonlinear FSDT Laminated Composite Plates

A Python implementation for exact solutions of linear and geometrically nonlinear (von Kármán) First-order Shear Deformation Theory (FSDT) laminated composite plates.

Based on the formulations and benchmark problems in:
> **J. N. Reddy**, *Mechanics of Laminated Composite Plates and Shells: Theory and Analysis*, CRC Press.

---

## Benchmark Results

### 1. Symmetric Cross-Ply `[0/90/90/0]`

#### Sinusoidally Distributed Load (SSL)
| $a/h$ | $\bar{w}$ | $\bar{\sigma}_{xx}$ | $\bar{\sigma}_{yy}$ | $\bar{\sigma}_{xy}$ | $\bar{\sigma}_{yz}$ | $\bar{\sigma}_{xz}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4   | 1.70951 | 0.40591 | 0.57643 | 0.03079 | 0.19618 | 0.34950 |
| 10  | 0.66271 | 0.49888 | 0.36142 | 0.02413 | 0.12918 | 0.41649 |
| 20  | 0.49117 | 0.52732 | 0.29565 | 0.02210 | 0.10868 | 0.43699 |
| 50  | 0.44095 | 0.53680 | 0.27373 | 0.02142 | 0.10185 | 0.44382 |
| 100 | 0.43368 | 0.53822 | 0.27045 | 0.02132 | 0.10083 | 0.44484 |

#### Uniformly Distributed Load (UDL)
| $a/h$ | $\bar{w}$ | $\bar{\sigma}_{xx}$ | $\bar{\sigma}_{yy}$ | $\bar{\sigma}_{xy}$ | $\bar{\sigma}_{yz}$ | $\bar{\sigma}_{xz}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4   | 2.56735 | 0.60783 | 0.83531 | 0.06134 | 0.46101 | 0.70274 |
| 10  | 1.02502 | 0.75782 | 0.50075 | 0.04722 | 0.36429 | 0.81301 |
| 20  | 0.76938 | 0.80455 | 0.39693 | 0.04218 | 0.33715 | 0.84488 |
| 50  | 0.69421 | 0.82045 | 0.36140 | 0.04012 | 0.32919 | 0.85492 |
| 100 | 0.68331 | 0.82285 | 0.35600 | 0.03973 | 0.32808 | 0.85643 |

---

### 2. Symmetric Cross-Ply `[0/90/0]`

#### Sinusoidally Distributed Load (SSL)
| $a/h$ | $\bar{w}$ | $\bar{\sigma}_{xx}$ | $\bar{\sigma}_{yy}$ | $\bar{\sigma}_{xy}$ | $\bar{\sigma}_{yz}$ | $\bar{\sigma}_{xz}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4   | 1.77575 | 0.43698 | 0.03068 | 0.03692 | 0.15609 | 0.36040 |
| 10  | 0.66930 | 0.51341 | 0.01764 | 0.02517 | 0.09145 | 0.40887 |
| 20  | 0.49214 | 0.53183 | 0.01450 | 0.02234 | 0.07588 | 0.42056 |
| 50  | 0.44106 | 0.53757 | 0.01353 | 0.02145 | 0.07102 | 0.42420 |
| 100 | 0.43370 | 0.53841 | 0.01338 | 0.02132 | 0.07031 | 0.42473 |

#### Uniformly Distributed Load (UDL)
| $a/h$ | $\bar{w}$ | $\bar{\sigma}_{xx}$ | $\bar{\sigma}_{yy}$ | $\bar{\sigma}_{xy}$ | $\bar{\sigma}_{yz}$ | $\bar{\sigma}_{xz}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4   | 2.65949 | 0.65456 | 0.04259 | 0.07369 | 0.42263 | 0.69430 |
| 10  | 1.02193 | 0.77193 | 0.02211 | 0.05159 | 0.32741 | 0.76755 |
| 20  | 0.75725 | 0.79837 | 0.01718 | 0.04547 | 0.30679 | 0.78239 |
| 50  | 0.68072 | 0.80612 | 0.01564 | 0.04318 | 0.30132 | 0.78663 |
| 100 | 0.66970 | 0.80722 | 0.01541 | 0.04277 | 0.30059 | 0.78725 |

---

### 3. Linear vs. Nonlinear Comparison (von Kármán, Single-Mode SSL)

#### Laminate: `[0/90/90/0]`
| $a/h$ | $w_{\text{lin}}$ | $w_{\text{NL}}$ | $\bar{\sigma}_{xx,\text{lin}}$ | $\bar{\sigma}_{xx,\text{NL}}$ | $\bar{\sigma}_{yy,\text{lin}}$ | $\bar{\sigma}_{yy,\text{NL}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4   | 1.70951 | 1.70951 | 0.40591 | 0.40566 | 0.57643 | 0.57618 |
| 10  | 0.66271 | 0.66264 | 0.49888 | 0.49736 | 0.36142 | 0.35992 |
| 20  | 0.49117 | 0.48550 | 0.52732 | 0.50868 | 0.29565 | 0.27968 |
| 50  | 0.44095 | 0.16046 | 0.53680 | 0.14178 | 0.27373 | 0.04605 |
| 100 | 0.43368 | 0.02872 | 0.53822 | 0.00819 | 0.27045 | -0.00955 |

#### Laminate: `[0/90/0]`
| $a/h$ | $w_{\text{lin}}$ | $w_{\text{NL}}$ | $\bar{\sigma}_{xx,\text{lin}}$ | $\bar{\sigma}_{xx,\text{NL}}$ | $\bar{\sigma}_{yy,\text{lin}}$ | $\bar{\sigma}_{yy,\text{NL}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4   | 1.77575 | 1.77575 | 0.43698 | 0.43670 | 0.03068 | 0.03066 |
| 10  | 0.66930 | 0.66922 | 0.51341 | 0.51182 | 0.01764 | 0.01757 |
| 20  | 0.49214 | 0.48643 | 0.53183 | 0.51270 | 0.01450 | 0.01373 |
| 50  | 0.44106 | 0.16048 | 0.53757 | 0.14051 | 0.01353 | 0.00236 |
| 100 | 0.43370 | 0.02872 | 0.53841 | 0.00742 | 0.01338 | -0.00042 |

---

## Nomenclature
- **$a/h$**: Side-to-thickness ratio
- **$\bar{w}$**: Non-dimensional center deflection
- **$\bar{\sigma}_{ij}$**: Non-dimensional stress components
- **SSL**: Sinusoidally Distributed Load
- **UDL**: Uniformly Distributed Load
- **lin / NL**: Linear vs. Nonlinear solution
