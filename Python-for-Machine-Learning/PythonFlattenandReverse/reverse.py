#reverse

my_input = [[1, 2], [3, 4], [5, 6, 7]]
my_output = []

def reverse(my_list):
    my_list=my_list[::-1]
    for i in my_list:
        if isinstance(i,list):
            my_list = i[::-1]
            my_output.append(my_list)
        else:
            my_output.append(my_list)
    return my_output
reverse(my_input)




