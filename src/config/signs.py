from pydantic_settings import BaseSettings


class SignConfigR(BaseSettings):
    """Sign configuration for R."""

    joint_wrj1: float = -0.000000
    joint_wrj0: float = -0.074235
    joint_ffj3: float = -0.165099
    joint_ffj2: float = 0.741008
    joint_ffj1: float = 0.009551
    joint_ffj0: float = 0.009950
    joint_mfj3: float = 0.317021
    joint_mfj2: float = 0.009179
    joint_mfj1: float = 0.009527
    joint_mfj0: float = 0.009936
    joint_rfj3: float = -0.339314
    joint_rfj2: float = 1.557509
    joint_rfj1: float = 1.498437
    joint_rfj0: float = 1.562332
    joint_lfj4: float = 0.213251
    joint_lfj3: float = -0.339351
    joint_lfj2: float = 1.561647
    joint_lfj1: float = 1.498414
    joint_lfj0: float = 1.562323
    joint_thj4: float = 0.718737
    joint_thj3: float = 1.071983
    joint_thj2: float = 0.199356
    joint_thj1: float = -0.503844
    joint_thj0: float = -1.561128
    action: list[float] = [
        0.555,
        0.176,
        -0.540,
        -0.070,
        -1.000,
        1.000,
        -1.000,
        -1.000,
        -1.000,
        1.000,
        1.000,
        -0.360,
        -1.000,
        1.000,
        1.000,
        0.710,
        0.750,
        1.000,
        -1.000,
        -1.000,
    ]


class SignConfigA(BaseSettings):
    """Sign configuration for A."""

    joint_wrj1: float = -0.000000
    joint_wrj0: float = -0.066764
    joint_ffj3: float = 0.000000
    joint_ffj2: float = 1.561403
    joint_ffj1: float = 0.885386
    joint_ffj0: float = 1.152955
    joint_mfj3: float = 0.000000
    joint_mfj2: float = 1.561407
    joint_mfj1: float = 0.995356
    joint_mfj0: float = 1.278506
    joint_rfj3: float = 0.000000
    joint_rfj2: float = 1.561407
    joint_rfj1: float = 1.042394
    joint_rfj0: float = 1.332207
    joint_lfj4: float = 0.073789
    joint_lfj3: float = -0.000256
    joint_lfj2: float = 1.561423
    joint_lfj1: float = 1.034548
    joint_lfj0: float = 1.323245
    joint_thj4: float = 0.059405
    joint_thj3: float = 0.493136
    joint_thj2: float = 0.199144
    joint_thj1: float = -0.514447
    joint_thj0: float = -0.456980
    action: list[float] = [
        0.555,
        0.176,
        0.000,
        1.000,
        0.120,
        0.000,
        1.000,
        0.260,
        0.000,
        1.000,
        0.320,
        -0.740,
        0.000,
        1.000,
        0.310,
        0.050,
        -0.160,
        1.000,
        -1.000,
        0.420,
    ]


class SignConfigI(BaseSettings):
    """Sign configuration for I."""

    joint_wrj1: float = -0.237400
    joint_wrj0: float = -0.070613
    joint_ffj3: float = 0.000000
    joint_ffj2: float = 1.561399
    joint_ffj1: float = 1.498428
    joint_ffj0: float = 1.562330
    joint_mfj3: float = 0.002116
    joint_mfj2: float = 1.561171
    joint_mfj1: float = 1.499321
    joint_mfj0: float = 1.562497
    joint_rfj3: float = 0.000000
    joint_rfj2: float = 1.561402
    joint_rfj1: float = 1.498437
    joint_rfj0: float = 1.562331
    joint_lfj4: float = 0.009137
    joint_lfj3: float = 0.000136
    joint_lfj2: float = 0.009362
    joint_lfj1: float = 0.009526
    joint_lfj0: float = 0.009936
    joint_thj4: float = 0.419010
    joint_thj3: float = 1.154544
    joint_thj2: float = 0.199643
    joint_thj1: float = -0.266863
    joint_thj0: float = -1.088588
    action: list[float] = [
        -0.200,
        0.176,
        0.000,
        1.000,
        1.000,
        0.000,
        1.000,
        1.000,
        0.000,
        1.000,
        1.000,
        -1.000,
        0.000,
        -1.000,
        -1.000,
        0.460,
        0.830,
        1.000,
        -0.590,
        -0.410,
    ]


class SignConfigL(BaseSettings):
    """Sign configuration for L."""

    joint_wrj1: float = -0.000000
    joint_wrj0: float = -0.070076
    joint_ffj3: float = -0.000000
    joint_ffj2: float = 0.009367
    joint_ffj1: float = 0.009529
    joint_ffj0: float = 0.009936
    joint_mfj3: float = 0.000000
    joint_mfj2: float = 1.561402
    joint_mfj1: float = 1.498437
    joint_mfj0: float = 1.562331
    joint_rfj3: float = 0.000000
    joint_rfj2: float = 1.561402
    joint_rfj1: float = 1.498437
    joint_rfj0: float = 1.562331
    joint_lfj4: float = 0.009268
    joint_lfj3: float = -0.000029
    joint_lfj2: float = 1.561407
    joint_lfj1: float = 1.498439
    joint_lfj0: float = 1.562331
    joint_thj4: float = -0.004823
    joint_thj3: float = 0.009317
    joint_thj2: float = 0.199055
    joint_thj1: float = 0.514358
    joint_thj0: float = -0.009639
    action: list[float] = [
        0.555,
        0.176,
        0.000,
        -1.000,
        -1.000,
        0.000,
        1.000,
        1.000,
        0.000,
        1.000,
        1.000,
        -1.000,
        0.000,
        1.000,
        1.000,
        0.000,
        -1.000,
        1.000,
        1.000,
        1.000,
    ]
