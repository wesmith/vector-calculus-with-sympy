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

aa = ws.coord_xforms()

def test_for_zero(data):
    dd = ws.Operators(data)
    if dd.divergence(dd.curl()) != 0: print('div(curl) test failed by {}'.format(dd.name))
    #assert dd.divergence(dd.curl()) == 0, 'div(curl) test failed by {}'.format(dd.name)
    if not dd.curl(dd.gradient())._eval_is_zero_matrix(): print('curl(grad) test failed by {}'.format(dd.name))
    #assert dd.curl(dd.gradient())._eval_is_zero_matrix(), 'curl(grad) test failed by {}'.format(dd.name)
    #print('"div(curl) and curl(grad)" tests passed by: {}'.format(dd.name))

def main():

    for key, val in aa.items():
        test_for_zero(val)

if __name__ == '__main__':

    main()
