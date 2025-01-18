Data from Horizons for Voyager 1, launched 1977-09-05 on board TC-6, *after* Voyager 2.

* `v1_horizons_elements.txt` -- Keplerian elements at 1-minute intervals from start of kernel to end of 1977-09-05.
* `v1_horizons_elements_1s.txt` -- Keplerian elements at 1-second intervals from start of kernel for 1 hour (3601 data points).
* `v1_horizons_vectors.txt` -- Earth-centered state vectors at same times as `v1_horizons_elements.txt`
* `v1_horizons_vectors_1s.txt` -- Earth-centered state vectors at same times as `v1_horizons_elements_1s.txt`

Earliest available data is at 1977-09-05T13:59:24.383 TDB, which is 13:58:36.200 UTC. This is 3755.242s (1h02m35.242s) after T-0. Notably,
the PM burnout is marked as being at T+3767.3, or 1h02m47.3s, 12 seconds after the first Horizons solution. The 1-second data
is there to see if the kernel records the end of PM burn.