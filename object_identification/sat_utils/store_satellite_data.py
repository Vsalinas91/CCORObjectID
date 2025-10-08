import numpy as np
from typing import Any


class AllCandidateSatelliteData:
    def __init__(self):
        self.get_sat_x = []
        self.get_sat_y = []
        self.get_proj_x = []
        self.get_proj_y = []
        self.sat_pix_x = []
        self.sat_pix_y = []
        self.get_sat_astro_x = []
        self.get_sat_astro_y = []
        self.get_sat_sep = []
        self.get_sat_name = []
        self.valid_angles = []
        self.pos_angles = []
        self.sat_x_error_min = []
        self.sat_x_error_max = []
        self.sat_y_error_min = []
        self.sat_y_error_max = []
        self.sat_x_mean = []
        self.sat_y_mean = []
        self.sat_tx = []
        self.sat_ty = []
        self.valid_time = []

    def build_sat_dict(self, j_times: list[Any], check_times: list[Any]) -> None:
        """
        Build the satellite dictionary object for plotting and writing to file.
        """
        sat_pix_x = np.array(self.get_sat_astro_x, dtype=object)
        sat_pix_y = np.array(self.get_sat_astro_y, dtype=object)
        sat_name_x = np.array(self.get_sat_name, dtype=object)
        sat_dists = np.array(self.get_sat_sep, dtype=object)
        sat_x_error_max = np.array(self.sat_x_error_max, dtype=object)
        sat_x_error_min = np.array(self.sat_x_error_min, dtype=object)
        sat_y_error_max = np.array(self.sat_y_error_max, dtype=object)
        sat_y_error_min = np.array(self.sat_y_error_max, dtype=object)
        sat_x_mean = np.array(self.sat_x_mean, dtype=object)
        sat_y_mean = np.array(self.sat_y_mean, dtype=object)
        sat_tx = np.array(self.sat_tx, dtype=object)
        sat_ty = np.array(self.sat_ty, dtype=object)
        valid_time = np.array(self.valid_time, dtype=object)

        candidate_sats = np.unique(np.concatenate(sat_name_x))

        self.sat_dict: dict[str, Any] = {}
        for csat in candidate_sats:
            name = []
            sat_xc = []
            sat_yc = []
            sat_dist = []
            sat_xe_max = []
            sat_xe_min = []
            sat_ye_max = []
            sat_ye_min = []
            sat_ye_mean = []
            sat_xe_mean = []
            sat_tx_val = []
            sat_ty_val = []
            time = []

            sub_dict = {}
            for idx in range(len(sat_pix_x)):
                sat_at_t = sat_name_x[idx]
                sat_x = sat_pix_x[idx]
                sat_y = sat_pix_y[idx]
                dist = sat_dists[idx]
                sat_x_max = sat_x_error_max[idx]
                sat_x_min = sat_x_error_min[idx]
                sat_y_max = sat_y_error_max[idx]
                sat_y_min = sat_y_error_min[idx]
                sat_x_mean_pix = sat_x_mean[idx]
                sat_y_mean_pix = sat_y_mean[idx]
                tx_val = sat_tx[idx]
                ty_val = sat_ty[idx]
                sat_time = valid_time[idx]

                if len(sat_at_t) != 0:
                    for enum, (vsat, x, y, dist) in enumerate(zip(sat_at_t, sat_x, sat_y, dist)):
                        if vsat == csat:
                            name.append(vsat)
                            sat_xc.append(x)
                            sat_yc.append(y)
                            sat_dist.append(dist)
                            # errors:
                            sat_xe_max.append(sat_x_max[enum])
                            sat_xe_min.append(sat_x_min[enum])
                            sat_ye_max.append(sat_y_max[enum])
                            sat_ye_min.append(sat_y_min[enum])
                            sat_xe_mean.append(sat_x_mean_pix[enum])
                            sat_ye_mean.append(sat_y_mean_pix[enum])
                            sat_tx_val.append(tx_val[enum])
                            sat_ty_val.append(ty_val[enum])
                            time.append(sat_time[enum])

            sub_dict["name"] = name
            sub_dict["x"] = sat_xc
            sub_dict["y"] = sat_yc
            sub_dict["dist"] = sat_dist
            sub_dict["x_max"] = sat_xe_max
            sub_dict["x_min"] = sat_xe_min
            sub_dict["y_max"] = sat_ye_max
            sub_dict["y_min"] = sat_ye_min
            sub_dict["x_mean"] = sat_xe_mean
            sub_dict["y_mean"] = sat_ye_mean
            sub_dict["tx"] = sat_tx_val
            sub_dict["ty"] = sat_ty_val
            sub_dict["sat_time"] = time

            self.sat_dict[csat] = sub_dict

        self.sat_dict["jtimes_searched"] = [j_times[i].tt for i in range(len(j_times))]
        self.sat_dict["times_searched"] = check_times
        self.sat_dict["solar_north_corrected"] = True
        self.sat_dict["tx_ty_unit"] = "deg"
        self.sat_dict["reference_frame"] = "helioprojective"


class SingleCandidateSatelliteData:
    def __init__(self):
        self.sat_coord_x = []
        self.sat_coord_y = []
        self.sat_name = []
        self.sat_angle_write = []
        self.pos_angle_write = []
        self.sat_sep = []
        self.sx_min_error = []
        self.sx_max_error = []
        self.sy_min_error = []
        self.sy_max_error = []
        self.sx_mean = []
        self.sy_mean = []
        self.tx_sat = []
        self.ty_sat = []
        self.time_to_sat = []
