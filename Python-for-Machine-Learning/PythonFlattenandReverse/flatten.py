#flatten

my_input = [[1,'a',['cat'],2],[[[3]],'dog'],4,5]
my_output = []

def flatten(my_list):
    for i in my_list:
        if isinstance(i,list):
            flatten(i)
        else:
            my_output.append(i)
    return my_output

flatten(my_input)
