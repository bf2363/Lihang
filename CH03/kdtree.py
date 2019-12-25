# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:18:56 2019

@author: bowei

KD tree and KNN
"""

class Node(object):
    
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return (not self.data) or \
                (all(not bool(c) for c,p in self.children))

    def preorder(self):
        if not self:
            return 
        
        yield self

        if self.left:
            for x in self.left.preorder():
                yield x
        
        if self.right:
            for x in self.right.preorder():
                yield x
    
    def inorder(self):
        if not self:
            return 

        if self.left:
            for x in self.left.inorder():
                yield x

        yield self

        if self.right:
            for x in self.right.inorder():
                yield x
        
    
    def postorder(self):
        if not self:
            return 
        
        if self.left:
            for x in self.left.postorder():
                yield x

        if self.right:
            for x in self.right.postorder():
                yield x

        yield self
    
    @property
    def children(self):
        if self.left and self.left.data is not None:
            yield self.left, 0

        if self.right and self.right.data is not None:
            yield self.right, 1

    def set_child(self, index,child):
        if index == 0:
            self.left = child
        else:
            self.right = child

    def height(self):
        min_height = int(bool(self))
        return max([min_height]+[c.height()+1 for c,p in self.children])
    
    def get_child_pos(self,child):
        for c,pos in self.children:
            if child == c:
                return pos
        

    def __repr__(self):
        return '<%(cls)s - %(data)s>' % \
            dict(cls=self.__class__.__name__, data=repr(self.data))

    def __nonzero__(self):
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self,other):
        if  isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return id(self)

    def require_axis(f):

        @wraps(f)
        def _wrapper(self,*args,**kwargs):
            if None in (self.axis, self.sel_axis):
                raise ValueError('%(func_name) requires the node %(node)s '
            'to have an axis and a sel_axis function' % dict(func_name=f.__name__, node=repr(self)))

            return f(self, *args, **kwargs)

        return _wrapper


class KDNode(Node):
    def __init__(self, data=None, left=None, right=None, axis=None,
                self_axis=None, dimensions=None):
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis= self_axis
        self.dimentions = dimensions
    
    @require_axis
    def add(self, point):
        
        current = self
        while True:
            if current.data is None:
                current.data = point
                return current
        
            if point[current.axis] < current.data[current.axis]:
                if current.left is None:
                    current.left = current.create_subnode(point)
                    return current.left
                else:
                    current = current.left
            
            else:
                if current.right is None:
                    current.right = current.create_subnode(point)
                    return current.right
                else:
                    current = current.right
    
    @require_axis
    def create_subnode(self, data):
        return self.__class__(data,
                            axis=self.sel_axis(self.axis),
                            sel_axis=self.sel_axis,
                            dimensions=self.dimentions)


    @require_axis
    
    

