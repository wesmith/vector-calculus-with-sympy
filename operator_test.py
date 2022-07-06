'''
MIT License

Copyright (c) 2022 Warren E Smith

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import operator_utils as ws
import unittest

aa = ws.Coords()
data = aa.spherical

class MyTest(unittest.TestCase):
    
    def test_for_zero(self):
        dd  = ws.Operators(data)
        val = dd.divergence(dd.curl())
        vec = dd.curl(dd.gradient())
        print(val, vec)
        with self.subTest():
            self.assertEqual(val, 0, "should be zero")
        with self.subTest():
            self.assertEqual(vec._eval_is_zero_matrix(), True, "should be zero vector")
    '''
    def test_cylindrical(self):
        dd = ws.Operators(aa.cylindrical)
        self.assertEqual(dd.divergence(dd.curl()), 0, "should be zero")
    def test_cartesian(self):
        dd = ws.Operators(aa.cartesian)
        self.assertEqual(dd.divergence(dd.curl()), 0, "should be zero")        
    '''
'''
class Multiple(MyTest):
    
    def __init__(self, aa):
        self.aa = aa

    def test_over_coords(self):
        self.test_for_zero(self.aa.spherical)
        self.test_for_zero(self.aa.cylindrical)
        self.test_for_zero(self.aa.cartesian)
'''

unittest.main()
