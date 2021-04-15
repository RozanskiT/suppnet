#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline

class InteractiveSpline:

    def __init__(self):
        self.knots_x= None
        self.knots_y= None

        self._spl= None

    def __call__(self,x):
        if self._spl is not None:
            return self._spl(x)
        else:
            return []
    
    def is_empty(self):
        return not bool(self.knots_x)

    def set_knots(self, knots_x, knots_y):
        if len(knots_x) >0:
            knots_x, knots_y= zip(*sorted(zip(knots_x, knots_y)))
        self.knots_x= list(knots_x)
        self.knots_y= list(knots_y)
        self.update_spline()

    def update_spline(self):
        if len(self.knots_x)>1:
            k= min(3,len(self.knots_x)-1)
            self._spl= InterpolatedUnivariateSpline(self.knots_x, self.knots_y, k=k)
        else:
            self._spl= None

    def add_knot(self,x,y):
        if len(self.knots_x) > 0:
            while True: # Never add overlapped point!
                idx= np.searchsorted(self.knots_x, x)
                idx= min(len(self.knots_x)-1,idx)
                if x != self.knots_x[idx]:
                    x += 0.001
                    break
            idx= np.searchsorted(self.knots_x, x)
            self.knots_x.insert(idx, x)
            self.knots_y.insert(idx, y)
        else:
            idx=0
            self.knots_x.append(x)
            self.knots_y.append(y)
        self.update_spline()
        return idx
        
    def replace_knot(self, new_x, new_y, old_idx):
        idx_rem= old_idx
        # idx_rem= min(len(self.knots_x)-1,old_idx)
        del self.knots_x[idx_rem]
        del self.knots_y[idx_rem]
        return self.add_knot(new_x, new_y)
        
    def get_knots(self):
        return self.knots_x, self.knots_y
    
    def remove_nearest_knot(self, x):
        if len(self.knots_x) != 0:
            idx= self.knots_x.index(min(self.knots_x, key=lambda xx:abs(xx-x)))
            del self.knots_x[idx]
            del self.knots_y[idx]
        self.update_spline()

def main():
    pass


if __name__ == '__main__':
	main()
