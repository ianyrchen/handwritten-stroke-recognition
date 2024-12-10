"""
Vectorizing yippee
Something not stupid
each stroke becomes  just displacement, curvature, 

O(n) stroke decomp
O(1)


"""

import threading
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
        return cls(points), len(points)
    
    def to_vectors(self, pts) -> np.ndarray:
        """
        Converts the sequence of points into vectors (x1, x2).
        
        Returns:
            A numpy array of shape (n, 2), where n is the number of points.
        """
        vectors = np.array([(int(x1), int(x2)) for x1, x2, t in pts])
        return vectors
    def stime(self, stroke):
        time = float(stroke[-1][2]) - float(stroke[0][2])
        return [(np.array([float(stroke[i][0]), float(stroke[i][1])]), (float(stroke[i][2]) - float(stroke[0][2]))) for i in range(len(stroke))]
    def alldstroke(self, tolerance, ex):
        strokes = []
        for st in self.points[ex]:
            strokes.append(self.dstroke(tolerance, st))
        #return [(self.bezier(3, strokes[i]) - (strokes[i-1][-1] if i !=0 else strokes[0][0])) if strokes[i] else None for i in range(len(strokes))]
        #print(strokes[0])
        #print([(strokes[i] - (strokes[i-1][-1] if i !=0 else strokes[0][0])) if strokes[i] else None for i in range(len(strokes))])

    def dstroke(self, tolerance, stroke):
        n = len(stroke)
        stroke = self.stime(stroke)
        del1 = [(stroke[i+1][0] - stroke[i][0])/(stroke[i+1][1] - stroke[i][1]) for i in range(n-1)]
        del2 = [del1[i+1] - del1[i] for i in range(n-2)]
        wacks = [0]
        print(del2)
        wacks.extend([i +1 for i, v in enumerate(del2) if np.linalg.norm(v) > tolerance])
        wacks.append(n-1)
        print(len(wacks)-2)
        return [stroke[a:b+1][0] for a,b in [(wacks[i], wacks[i+1]) for i in range(len(wacks)-1)] if b-a > 3]

    def nflat(self, n, stroke):
        k = len(stroke)
        return self.to_vectors([stroke[round((k-1)*i/(n-1))] for i in range(n)])
    def nflatall(self, n, ex):
        return [self.nflat(n, self.points[ex][stroke]) for stroke in range(len(self.points[ex]))]
            
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
    def postproc(self, ex, n):
        bob = self.nflatall(n, ex)
        return [[bob[s][i] - bob[max(0, s-1)][-1] for i in range(len(bob[s]))] for s in range(len(bob))]
        """rets = []
        for i in range(len(ex)):
            b = ex[i] - ex[max(0,i-1)][-1]
            #print(b.shape)
            #print(ex[i].shape)
            rets.append(self.nflatall(d, ex[i] - ex[max(0,i-1)][-1]))
        return rets
"""
# Example usage:
# Assuming you have a .pkl file named 'data.pkl' with points (x1, x2, t)
if __name__ == "__main__":
#def mainloop():
    #print("vecting")
    fname = 'x_data.pkl'
    #fname = 'whiteboardtest.pkl'


    processor, size = Vectorize.from_pkl(fname)
    
    all_data = []

    ### random bullshit time
    bob = size
    for ex in range(bob):
        print(ex)
        x = processor.alldstroke(100, ex)
        #print(x[ex].shape)
        # 47 is the minimum number of strokes for any given datapoint
        # x is a list of strokes
        #ctrlx = [processor.bezier(4, x[i]) for i in range(len(x))]
        # ctrlx = processor.preproc(x, 4)
        # 4 DOF -> 5 size dimension
        # resulting ctrlx is 47 by 5 by 2, representing 47 strokes of 5 (x,y) control points

        all_data.append(x)
    #savewb = 'bezwb.pkl'
    savewb = 'x_kill_data.pkl'
    with open(savewb, 'wb') as file:
        pickle.dump(all_data, file)
        file.close()
#    threading.Timer(0.5, mainloop).start()
#mainloop() 
