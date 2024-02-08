import numpy as np


class LOSGuidance:
    #def __init__(self, p0: list[float], p1: list[float]):
        #self.set_path(p0, p1)
        #self.heading_ref = 50*np.pi/180 # magic init number!!!
        #self.p_next = [np.array([50, -40]), np.array([20.0, -80.0]), np.array([120.0, -60.0]), np.array([160, 0.0]), np.array([60.0, 60.0])]
        #self.acceptance_radius = 0.5
    def __init__(self, p_ref, p_target, asv_pos):
        self.p_ref = p_ref
        self.p_target = p_target
        self.p = asv_pos
        self.pi_p = self.calculate_pi_p()
        self.y_e = self.cross_track()
        #self.R_Pi_p = self.calculate_R_Pi_p()

    # def set_path(self, p0: list[float], p1: list[float]):
    #     self.p0 = np.array(p0)
    #     self.p1 = np.array(p1)
    def calculate_pi_p(self):
        Pi_p = np.arctan2(self.p_target[1]-self.p_ref[1], self.p_target[0]-self.p_ref[0])
        return Pi_p


    def cross_track(self):
        # Equation 12.43 in Fossen
        A = np.array(
            [[np.cos(self.Pi_p), np.sin(self.Pi_p), 0],
            [-np.sin(self.Pi_p), np.cos(self.Pi_p), 1],
            [np.tan(self.Pi_p), -1, 0]]
        )
        B = np.array(
            [[np.cos(self.Pi_p) * self.p[0] + np.sin(self.Pi_p) * self.p[1]],
            [-np.sin(self.Pi_p) * self.p[0] + np.cos(self.Pi_p) * self.p[1]],
            [np.tan(self.Pi_p) * self.p_target[0] - self.p_target[1]]]
        )

        x = np.linalg.inv(A) @ B
        y_e = x[2]

        return y_e

    # def calculate_distance(self, p0, p1):
    #     return np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
    
    # def switch_path(self):
    #     self.p0, self.p1 = self.p1, self.p_next[0]

    def calculate_psi_d(self, look_ahead: float):
        # TODO FIX this, return psi_d = chi_d = Pi_p - arctan2(1/look_ahead * )
        # page 342 in Fossen
        chi_d = self.pi_p - np.arctan(self.y_e/look_ahead)
        
        # Ignoring "problem"

        psi_d = chi_d

        return psi_d
        
    def calculate_LOS_x_ref(self, x: np.ndarray, look_ahead: float) -> np.ndarray:
        self.set_path(self.p0, self.p1)
        self.calculate_R_Pi_p()
        p_asv = np.array([x[0], x[1]])
        errors = self.R_Pi_p.T @ (p_asv - self.p0)
        along_track_error = errors[0]
        
        # Switch points to next pair if crossed orthogonal line or entered circle of acceptance
        if ((along_track_error > 0.8*self.calculate_distance(self.p0, self.p1))):
            self.switch_path()
            self.calculate_R_Pi_p()
            errors = self.R_Pi_p.T @ (p_asv - self.p0)
            along_track_error = errors[0]
            self.p_next.pop(0)
            if (self.p_next == []):
                self.p_next = [np.array([100000.0, 1000000.0])]
            

        p_los_world = self.R_Pi_p @ np.array([along_track_error + look_ahead, 0]) + self.p0
        x_ref = np.array([p_los_world[0], p_los_world[1]])
        
        return x_ref
    
