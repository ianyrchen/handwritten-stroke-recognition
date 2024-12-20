"""
Vectorizing yippee

"""


import pickle
import numpy as np
from scipy.special import comb as nOk
from typing import List, Tuple


Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*nOk(n,k)
bezierM = lambda ts: np.matrix([[Mtk(3,t,k) for k in range(4)] for t in ts])
def lsqfit(points,M):
    M_ = np.linalg.pinv(M)
    return M_ * points


class Vectorize:
    def __init__(self, points: List[List[Tuple[float, float, float]]]):
        """
        Initializes the processor with a list of points.
        Each point is a tuple (x1, x2, t).
        
        Args:
            points: matrix, where each tuple contains (x1, x2, t)
        """
        self.points = points    
    @classmethod
    def from_pkl(cls, file_path: str):
        """
        Loads points from a .pkl file and creates an instance of the class.
        
        Args:
            file_path: Path to the .pkl file.
        
        Returns:
            PointSequenceProcessor instance with loaded points.
        """
        with open(file_path, 'rb') as file:
            points = pickle.load(file)
        return cls(points)
    
    def to_vectors(self, pts) -> np.ndarray:
        """
        Converts the sequence of points into vectors (x1, x2).
        
        Returns:
            A numpy array of shape (n, 2), where n is the number of points.
        """
        vectors = np.array([(int(x1), int(x2)) for x1, x2, t in pts])
        return vectors

    def getij(self, i,j):
        return self.to_vectors(self.points[i][j])
    def getex(self,ex):
        return [self.to_vectors(self.points[ex][i]) for i in range(len(self.points[ex]))]
    def bezier(self, deg, pts):
        """
        Finds least square fit bezier curve of degree deg, and returns the control points

        Input: a sequence of points [(x1, x2)]

        """

        # Bernstein polynomial
        Mtk = lambda n, t, k: t**(k)*(1-t)**(n-k)*nOk(n,k)
        # interpolate at ts
        bezierM = lambda ts: np.matrix([[Mtk(deg,t,k) for k in range(deg+1)] for t in ts])
        def lsqfit(points,M):
            M_ = np.linalg.pinv(M)
            return M_ * points
        T = np.linspace(0, 1, len(pts))
        control_pts = lsqfit(pts,bezierM(T))
        control_pts[0] = pts[0]
        control_pts[-1] = pts[-1]
        return control_pts


    

# assume you have a .pkl file named 'data.pkl' with points (x1, x2, t)
if __name__ == "__main__":
    processor = Vectorize.from_pkl('x_data.pkl')
    
    all_bezier_data = []

    for ex in range(962):
        print(ex)
        x = processor.getex(ex)
        # 47 is the minimum number of strokes for any given datapoint
        # x is a list of strokes
        ctrlx = [processor.bezier(4, x[i]) for i in range(len(x))]
        # 4 DOF -> 5 size dimension
        # resulting ctrlx is 47 by 5 by 2, representing 47 strokes of 5 (x,y) control points

        all_bezier_data.append(ctrlx)

    with open('x_bezier_data.pkl', 'wb') as file:
        pickle.dump(all_bezier_data, file)
        file.close()

