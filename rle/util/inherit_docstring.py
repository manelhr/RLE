import types
import inspect
import re
from collections import OrderedDict


def _get_dic_docstring(docs):

    vlist = docs.split(":return:")
    vlist[0] = vlist[0].split(":param")
    vlist = vlist[0] + [vlist[1]]
    dic = OrderedDict()
    dic["main"] = re.sub("(\t|\n|  )+", " ", vlist[0].strip())
    dic["return"] = re.sub("(\t|\n|  )+", " ", vlist[-1].strip())
    for attribute in vlist[1:-1]:
        tmp = attribute.split(":")
        dic[tmp[0].strip()] = re.sub("(\t|\n|  )+", " ", tmp[1].strip())

    return dic


def inherit_docstring(cls):

    for name, func in vars(cls).items():

        if isinstance(func, types.FunctionType):

            doc_func = getattr(func, '__doc__', None)

            if doc_func is not None:
                child_dic = _get_dic_docstring(doc_func)

            for parent in inspect.getmro(cls)[1:-1]:

                parent_func = getattr(parent, name, None)
                doc_parent_func = getattr(parent_func, '__doc__', None)

                # if there is no definition on the child and there is on a parent, get the first parent
                if doc_func is None \
                        and parent_func is not None \
                        and doc_parent_func is not None:
                    func.__doc__ = parent_func.__doc__

                elif doc_func is not None \
                        and parent_func is not None \
                        and doc_parent_func is not None:

                    parent_dic = _get_dic_docstring(doc_parent_func)

                    for key in child_dic.keys():
                        if key in parent_dic.keys():
                            child_dic[key] = re.sub("defined@" + parent.__name__, parent_dic[key], child_dic[key])

                    doc_v = "\n" + child_dic["main"] + "\n\n"
                    for key in child_dic.keys():
                        if key != "main" and key != "return":
                            doc_v += ":param " + key + ": " + child_dic[key] + "\n"
                    doc_v += ":return: " + child_dic["return"]

                    func.__doc__ = doc_v

    return cls

