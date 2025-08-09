
class Lambda(object):
    @staticmethod
    def reduce_sum():
        def s(x,y):
            return x+y;
        return s;

    @staticmethod
    def reduce_product():
        def p(x,y):
            return x*y;
        return p;

    @staticmethod
    def map_attribute(attribute_name):
        def a(x):
            return getattr(x, attribute_name);
        return a;
