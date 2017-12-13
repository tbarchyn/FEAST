mcfpd_to_gps = 1000 / 24 / 60 * 0.0283 / 60 * 1e5 / 8.314 / 293 * 16
kgph_to_gps = 1000 / 3600

OGI_DIR_NAME = "ogi_baseline"

TECH_NAME = "AD"
BASELINE_NAME = "MIR"

null_repair_rate_per_day = 0.1 / 360
    # assumes a 10% chance that you'll find the leak when
    # you visit the site, once a year.. This may be too high. However, in principle this
    # should cancel out when we difference Leak Surveyor against OGI.