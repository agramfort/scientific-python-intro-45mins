Intro to scientific Python in 45'
================================================================================

----

... or Python for Matlab Users
================================================================================

----

What you should be able to do
--------------------------------------------------------------------------------

... in 45mins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- start Python
- do simple math
- basic scripting
- get started with linear algebra and scientific computing
- plotting

----

What use Python for?
--------------------------------------------------------------------------------

- scripting (like shell scripts e.g. bash, csh)
- make web sites (like these slides)
- **science** (like Matlab, IDL, R, Octave, Scilab)
- etc.

You just need to know 1 language to do almost everything !

----

Scientific Python building blocks
-----------------------------------

* **Python**, a generic and modern computing language

* **IPython**, an advanced **Python shell**: http://ipython.org/

* **Numpy** : provides powerful **numerical arrays** objects, and routines to
  manipulate them: http://www.numpy.org/

* **Scipy** : high-level data processing routines.
  Optimization, regression, interpolation, etc: http://www.scipy.org/

* **Matplotlib** a.k.a. Pylab: 2-D visualization, "publication-ready" plots
  http://matplotlib.sourceforge.net/

* **Mayavi** : 3-D visualization
  http://code.enthought.com/projects/mayavi/

----

First step
--------------------------------------------------------------------------------

Start the **Ipython** shell (from terminal or Windows cmd shell):

.. sourcecode:: bash

    $ ipython -pylab

Getting a scientific-Python environment:

* Comes with every Linux distribution
* Python(x,y) on Windows: http://www.pythonxy.com
* EPD: http://www.enthought.com/products/epd.php

----

Hello world!
--------------------------------------------------------------------------------

Start IPython:

.. sourcecode:: bash

    $ ipython -pylab

.. raw:: html

  <span class="pylab_demo">

.. image:: images/snapshot_ipython.png
  :scale: 60%

.. raw:: html

  </span>

Once you have started the interpreter, type:

.. sourcecode:: python

    >>> print "Hello, world!"
    Hello, world!

----

Python basics: Numerical types
--------------------------------------------------------------------------------

Integer variables:

.. sourcecode:: python

    >>> 1 + 1
    2
    >>> a = 4

floats:

.. sourcecode:: python

    >>> c = 2.1

complex (a native type in Python!):

.. sourcecode:: python

    >>> a = 1.5 + 0.5j
    >>> a.real
    1.5
    >>> a.imag
    0.5

----

Python basics: Numerical types
--------------------------------------------------------------------------------

and booleans:

.. sourcecode:: python

    >>> 3 < 4
    True
    >>> test = (3 > 4)
    >>> test
    False
    >>> type(test)
    <type 'bool'>

Note that **you don't need to specify the type** of the variable

.. sourcecode:: C

    int a = 1;  # in C

----

Python basics: Numerical types
--------------------------------------------------------------------------------

Python can replace your pocket calculator with : ``+``, ``-``, ``*``, ``/``, ``%`` (modulo)

.. sourcecode:: python

    >>> 7 * 3.
    21.0
    >>> 2**10
    1024
    >>> 8 % 3
    2

**WARNING** : Integer division

.. sourcecode:: python

    >>> 3 / 2  # !!!
    1
    >>> 3 / 2.  # Trick: use floats
    1.5
    >>> 3 / float(2)  # type conversion
    1.5

----

Python basics: container types
--------------------------------------------------------------------------------

The *list* type:

.. sourcecode:: python

    >>> a = [1]

Or

.. sourcecode:: python

    >>> a = list()
    >>> a.append(1)
    [1]

Concatenation and access:

.. sourcecode:: python

    >>> a + a  # concatenation
    [1, 1]
    >>> a[0] = 2  # access 1st element (starts at 0!)
    [2, 1]
    >>> a[-1] = 0  # access last element
    [2, 0]


----

Python basics: container types
--------------------------------------------------------------------------------

* Slicing: obtaining sublists of regularly-spaced elements

.. sourcecode:: python

    >>> l = [1, 2, 3, 4, 5]
    >>> l[2:4]
    [3, 4]

Note that i is in ``l[start:stop]`` if ``start <= i < stop``

**Slicing syntax**: `l[start:stop:stride]`

.. sourcecode:: python

    >>> l[:3]  # first 3
    [1, 2, 3]
    >>> l[3:]  # from 3 to end
    [4, 5]
    >>> l[::2]
    [1, 3, 5]

----

Python basics: container types
--------------------------------------------------------------------------------

Reverse `l`:

.. sourcecode:: python

    >>> r = l[::-1]
    >>> r
    [5, 4, 3, 2, 1]

Sort (in-place):

.. sourcecode:: python

    >>> r.sort()
    >>> r
    [1, 2, 3, 4, 5]


``r.sort()`` or ``r.append(1)`` are examples of object-oriented programming (OOP).
Being a ``list``, the object `r` owns the *method* `function` that is called
using the notation **.**

That's all you need to know today.

----

Python basics: data types
--------------------------------------------------------------------------------

Strings: *str*

.. sourcecode:: python

    >>> a = "hello, world!"
    >>> print a[2]
    'l'
    >>> a.replace('l', 'z', 1)
    'hezlo, world!'
    >>> a.replace('l', 'z')
    'hezzo, worzd!'

* String substitution:

.. sourcecode:: python

    >>> 'An integer: %i; a float: %f; a string: %s' % (1, 0.1, 'string')
    'An integer: 1; a float: 0.100000; another string: string'

Behaves very much like printf in C

.. sourcecode:: python

    >>> print "%03d" % 2  # print fixed size
    "002"

----

Python basics: data types
--------------------------------------------------------------------------------

A dictionary ``dict`` is basically an efficient table that **maps keys to
values**. It is an **unordered** container:

.. sourcecode:: python

    >>> phone = {'ellen': 5752, 'khaldoun': 5578}
    >>> phone['alex'] = 5915
    >>> phone
    {'khaldoun': 5578, 'alex': 5915, 'ellen': 5752}  # no order
    >>> phone['sebastian']
    5578
    >>> phone.keys()
    ['khaldoun', 'alex', 'ellen']
    >>> phone.values()
    [5578, 5915, 5752]
    >>> 'ellen' in phone
    True


----

Getting help
--------------------------------------------------------------------------------

Start `ipython`:

.. sourcecode:: python

    >>> print('Hello world')
    Hello world
    >>> print?  # don't forget the ?
    Type:		builtin_function_or_method
    Base Class:	        <type 'builtin_function_or_method'>
    String Form:	<built-in function print>
    Namespace:	        Python builtin
    Docstring:
	print(value, ..., sep=' ', end='\n', file=sys.stdout)

	Prints the values to a stream, or to sys.stdout by default.
	Optional keyword arguments:
	file: a file-like object (stream); defaults to the current sys.stdout.
	sep:  string inserted between values, default a space.
	end:  string appended after the last value, default a newline.


-----

Numpy
--------------------------------------------------------------------------------

**Numpy** is:

    - an extension package to Python for multidimensional arrays (matrices in n-dimensions)

    - designed for **efficient** scientific computation

Example:

.. sourcecode:: python

     >>> import numpy as np
     >>> a = np.array([0, 1, 2, 3])
     >>> a
     array([0, 1, 2, 3])

Reference documentation: http://docs.scipy.org


-----

Numpy: Creating arrays
--------------------------------------------------------------------------------

* 1-D

.. sourcecode:: python

    >>> a = np.array([0, 1, 2, 3])
    >>> a
    array([0, 1, 2, 3])

Getting the size and dimensions of the array:

.. sourcecode:: python

    >>> a.ndim
    1
    >>> a.shape
    (4,)
    >>> len(a)
    4

-----

Numpy: Creating arrays
--------------------------------------------------------------------------------

* 2-D

.. sourcecode:: python

    >>> b = np.array([[0, 1, 2], [3, 4, 5]])    # 2 x 3 array
    >>> b
    array([[ 0,  1,  2],
           [ 3,  4,  5]])
    >>> b.ndim
    2
    >>> b.shape
    (2, 3)
    >>> len(b)     # returns the size of the first dimension
    2

* 3-D, ...

.. sourcecode:: python

    >>> c = np.array([[[1], [2]], [[3], [4]]])
    >>> c.shape
    (2, 2, 1)

.. In practice, we rarely enter items one by one...

-----

Numpy: Creating arrays
--------------------------------------------------------------------------------

* Evenly spaced:

.. sourcecode:: python

    >>> import numpy as np
    >>> a = np.arange(10) # 0 .. n-1  (!)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = np.arange(1, 9, 2) # start, end (exlusive), step
    >>> b
    array([1, 3, 5, 7])

* or by number of points:

.. sourcecode:: python

    >>> c = np.linspace(0, 1, 6)   # start, end, num-points
    >>> c
    array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])

-----

Numpy: Creating arrays
--------------------------------------------------------------------------------

* Common arrays: **ones**, **zeros** and **eye** (like in Matlab)

.. sourcecode:: python

    >>> a = np.ones((3, 3))
    >>> a
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])

.. sourcecode:: python

    >>> b = np.zeros((2, 2))
    >>> b
    array([[ 0.,  0.],
           [ 0.,  0.]])

.. sourcecode:: python

    >>> c = np.eye(3)
    >>> c
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

-----

Numpy: Creating arrays
--------------------------------------------------------------------------------

* Random numbers:

.. sourcecode:: python

    >>> a = np.random.rand(4)              # uniform in [0, 1]
    >>> a
    array([ 0.58597729,  0.86110455,  0.9401114 ,  0.54264348])
    >>> b = np.random.randn(4)             # gaussian
    >>> b
    array([-2.56844807,  0.06798064, -0.36823781,  0.86966886])

In n-dimensions:

.. sourcecode:: python

    >>> c = np.random.rand(3, 3)
    >>> c
    array([[ 0.31976645,  0.64807526,  0.74770801],
           [ 0.8280203 ,  0.8669403 ,  0.07663683],
           [ 0.11527489,  0.11494884,  0.13503285]])

-----

Numpy: Basic data types
--------------------------------------------------------------------------------

.. sourcecode:: python

    >>> a = np.array([1, 2, 3])
    >>> a.dtype
    dtype('int64')

has a **different data type** than:

.. sourcecode:: python

    >>> b = np.array([1., 2., 3.])
    >>> b.dtype
    dtype('float64')

You can also choose:

.. sourcecode:: python

    >>> c = np.array([1, 2, 3], dtype=float)
    >>> c.dtype
    dtype('float64')

**Remark:** Much of the time you don't necessarily need to care, but remember they are there.

.. Remark: There are also other types (e.g. 'complex128', 'bool', etc.)

-----

Visualization with Python
--------------------------------------------------------------------------------

.. sourcecode:: python

    >>> import pylab as pl
    >>> t = np.linspace(0, 8 * np.pi, 1000)
    >>> x = np.sin(t)
    >>> pl.plot(t, x)
    >>> pl.xlabel('Time')
    >>> pl.ylabel('Amplitude')
    >>> pl.ylim([-1.5, 1.5])
    >>> pl.show()
    >>> pl.savefig('pylab_demo.pdf')  # natively save pdf, svg, png etc.

.. raw:: html

  <span class="pylab_demo">

.. image:: images/pylab_demo.png
  :scale: 45%

.. raw:: html

  </span>

-----

Visualization with Python
--------------------------------------------------------------------------------

* 2-D (such as images)

.. sourcecode:: python

    >>> image = np.random.rand(30, 30)
    >>> pl.imshow(image)
    >>> pl.gray()
    >>> pl.show()

.. raw:: html

  <span class="pylab_demo">

.. image:: images/pylab_image_demo.png
  :scale: 45%

.. raw:: html

  </span>

-----

Visualization with Python
--------------------------------------------------------------------------------

* 3-D with Mayavi

.. raw:: html

  <span class="pylab_demo">

.. image:: images/plot_fmri_contours.png
  :scale: 95%

.. raw:: html

  </span>

Check out: http://pysurfer.github.com/

-----

My first script
--------------------------------------------------------------------------------

.. sourcecode:: ipython

    In [3]: %run my_script.py
    Hello word

    In [4]: s
    Out[4]: 'Hello word'

    In [5]: %whos
    Variable   Type    Data/Info
    ----------------------------
    s          str     Hello word


-----

Getting started at the Martinos
--------------------------------------------------------------------------------

... tomorrow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a terminal do:

.. sourcecode:: tcsh

    $ setenv PATH /usr/pubsw/packages/python/epd/bin:${PATH}

If you use Bash replace the previous instruction with:

.. sourcecode:: bash

    $ export PATH=/usr/pubsw/packages/python/epd/bin:${PATH}

Then start the python interpreter with:

.. sourcecode:: bash

    $ ipython -pylab

-----

Learn more
--------------------------------------------------------------------------------

- http://scipy-lectures.github.com
- http://www.scipy.org/NumPy_for_Matlab_Users

For a Matlab like IDE environment

- http://packages.python.org/spyder

Parallel computing:

- http://packages.python.org/joblib

MEG and EEG data analysis:

- http://martinos.org/mne
