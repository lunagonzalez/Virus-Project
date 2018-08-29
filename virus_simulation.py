# Some background: In this code, I represent each virus as a string of 0, 1, or 2 of length 13000 (~the length of the genome)
# I store these viruses in a balancing binary tree structure. The nodes are sorted based on the sum of the numbers in their genome.
# Each node of the tree corresponds to a particular population of viruses who have exactly the same genome. That way, I don't store an unnecessary number of nodes in the tree.
# That is also where the size_next and size_now elements come from, size_next is the number of viruses in the current population with that exact genome, and size_next is the number of
#children they have. 
# At each generation, I travel to each node in the tree, and draw a number of children who have 0, 1, 2,...,10 mutations, and then give its children that random number of mutations.
# I create a new tree, global_tree, the global variable defined below, and add each of these children to it to get the next generation.
# Then, at the end of 8 generations, I do a bottleneck back to 200 viruses, where I randomly draw 200 individuals from the tree to keep into the next generation.

# The first few functions define distributions I use to get the number of children and the number of mutation they have. These distributions come from the arxiv paper I posted.
# The next functions define the node class, and define tree operations. kd_tree makes a perfectly balanced binary tree given a specific set of nodes, which is useful in starting with P0.
# The rest of the functions are fairly self-explanatory with their comments (I hope), except whole_step_2() traverses the tree, addingthe next generations children to the new tree,
# global_tree, and step_3_and_4() just sets the current tree to be this new generation, and sets all the size_next values to be the new size_now values. At the end of the bottleneck, you
# can use the make_freq_vector function to get a vector of mutation frequencies in the population.

#All the things at the end are my test code.

# The error is occuring in the add to tree function specifically in the tree rotations. One issue is that heights in the tree aren't correctly reported, but another is that when nodes are
#rotated, for some reason some nodes get deleted from the tree. However, the simple test cases I ran for the tree roations (at the bottom of the main code) work well, so I'm not sure what
#the issue is. You can see the pdf of class notes I attached for information on how self-balancing binary trees are supposed to work.

import math
import random
import numpy
import datetime
import sys

##some constants for our population and some math functions for drawing from distributions
#per site mut rate
Mu=10**(-4)
#length of genome
r=13000
#growth rate of population
alpha=2.175
#max number of new mutations a single virus can get per generation
d_max=10
# bottleneck size, should be set to 200 eventually
bottleneck_size=200
# Have to set an empty new tree in the beginning of each generation.
# Will also need to set global_tree to be empty at the beginning of each generation.
# this variable is also used for doing the bottleneck at the end of a passage.
global_tree=None
# Set an empty population size, which is recounted at the end of every passage.
pop_size=0
# counter is used in the bottleneck function. Starts at -1 because choosing in range pop_size includes the index 0.
#this is also used to count what virus you're on when traversing the tree to add things to freq_count_mat
counter=-1
#this second counter is used to count which number virus we are pulling from the population when doing the bottleneck.
counter2=0
# this is used to count the frequencies at the end of one passage. Then we only need one, so it's the global variable freq_count_mat
class freq_counter():
    def __init__(self, rows):
        self.rows=rows
        self.matrix=[[0 for i in range(r)] for j in range(rows)]
    def set_row_num(self, new_num): #this can also be used to reset the matrix to something small to decrease memory usage.
        self.rows=new_num
        self.matrix=[[0 for i in range(r)] for j in range(new_num)]
#initializing it with one row, but need to set rows=pop_size somewhere in the main code. I do this inside bottleneck which calls get_new_pop.
freq_count_mat=freq_counter(1)

#choose j elements out of n
def choose(n,j):
    return math.factorial(n)/(math.factorial(n-j)*math.factorial(j))

#change of d mutations occuring in child
def Nu_d(d):
    return choose(13000,d)*(Mu**d)*((1-Mu)**(13000-d))

# this function calculates the number of children a virus (or population of viruses with identical mutations) produces with a certain number of mutations. So the first component
#is number of children with 0 mutations, second is number of children with 1 mutation, and so on, where the last component is number of children with d_max mutations.
def calc_z_d(node):
    next_gen=[]
    for d in range(d_max+1):
        next_gen.append(numpy.random.poisson(alpha*Nu_d(d)*node.size_now))
    return next_gen

#this is what each element in the tree is. Each element represents a population of viruses that all have identical mutations.
#The attribute mut_key gives the "genome" of this population which a list of length 13000, where a 1 is WT and a 0 or 2 is a mutations.
#size_now and size_next count the number of viruses with exactly this mutational make up in the current generation and in the next generation.
#left, right, and parent are the left and right children and parent of the node in the tree. The parent of the root node will just be None.
#value is calculated by taking the mutation key as a number base 3. This is the value used to sort the elements in the tree.
# the height of a node is the maximum height of either of its children plus 1.
class Node():

    def __init__(self, mut_key, s, size_now, size_next, left, right, parent):
        stringrep=''.join(map(str,mut_key))
        self.value = int(stringrep,3)
        self.left = left
        self.right = right
        self.mut_key = mut_key
        self.s=s
        self.size_now=size_now
        self.size_next=size_next
        self.parent=parent
        if self.left is None and self.right is None:
            self.height=0
        elif self.left is None and self.right is not None:
            self.height=self.right.height+1
        elif self.left is not None and self.right is None:
            self.height=self.left.height+1
        else:
            self.height=max(self.left.height, self.right.height)+1

    def getLeftChild(self):
        return self.left
    def getRightChild(self):
        return self.right
    def getNodeValue(self):
        return self.value
    def setMutKey(self,new_key): #this function also updates the value used for sorting
        self.mut_key = new_key
        self.value = sum(new_key)
    def getMutKey(self):
        return self.mut_key
    def setSizeNow(self,new_size):
        self.size_now=new_size
    def setSizeNext(self,new_size):
        self.size_next=new_size
    def getSizeNow(self):
        return self.size_now
    def getSizeNext(self):
        return self.size_next
    def set_next_now(self):
        self.size_now=self.size_next
        self.size_next=0
    #PrintNicely is a way to print the tree. It prints across rows, so it will say which row it is, and then all the information below about the nodes. Useful for debugging.
    def PrintNicely(self):
        done=0
        row_num=0
        row=[self]
        while (not done):
            check_done=0
            print("row number", row_num)
            for i in range(2**row_num):
                if row[i] is None:
                    print("empty")
                    check_done+=1
                else:
                    if row[i].parent is None:
                        parent_info='root node'
                    else:
                        parent_info=row[i].parent.mut_key
                    print(row[i].mut_key, " size next:", row[i].size_next, " size now:", row[i].size_now, " value key:", row[i].value, " height:", row[i].height, " parent mut key:", parent_info) 
            new_row=[0 for i in range(2**(row_num+1))]
            for j in range(2**row_num):
                if row[j] is not None:
                    new_row[j*2]=row[j].left
                    new_row[j*2+1]=row[j].right
                else:
                    new_row[j*2]=None
                    new_row[j*2+1]=None
            row=new_row
            if check_done==2**row_num:
                done=1
            row_num+=1


#this function makes a tree given a specific set of starting viruses. This is only used when initializing the population at P0.
def kdtree(mut_list, s_list, size_now_list, size_next_list):
    try:
        k = len(mut_list[0]) # assumes all points have the same dimension
    except IndexError as e: # if not point_list:
        return None
    
    ## Sort point list as well as other attributes and choose median as pivot element
    #create permutation list that tells you how to sort s_list, size_now_list, and size_next_list in the same way as the mutation keys.
    size_permute=len(mut_list)
    sum_keys=[int(''.join(map(str, mut_list[i])),3) for i in range (size_permute)]
    permutation=[i for i in range(size_permute)]
    keydict = dict(zip(permutation, sum_keys))
    permutation.sort(key=keydict.get)
    #actually sorting.
    s_sort=[0 for i in range(size_permute)]
    now_sort=[0 for i in range(size_permute)]
    next_sort=[0 for i in range(size_permute)]
    mut_sort=[[] for i in range(size_permute)]
    for i in range(size_permute):
        s_sort[i]=s_list[permutation[i]]
        now_sort[i]=size_now_list[permutation[i]]
        next_sort[i]=size_next_list[permutation[i]]
        mut_sort[i]=mut_list[permutation[i]]
    s_list=s_sort
    size_now_list=now_sort
    size_next_list=next_sort
    mut_list=mut_sort
    median = len(sum_keys) // 2 # choose median

    # Create node and construct subtrees
    return Node(mut_list[median],
                s_list[median],
                size_now_list[median],
                size_next_list[median],
                kdtree(mut_list[:median], s_list[:median], size_now_list[:median], size_next_list[:median]),
                kdtree(mut_list[median+1:], s_list[median+1:], size_now_list[median+1:], size_next_list[median+1:]),
                parent=None
                )

#this is to add the parents to a tree made with kdtree since it's recursive and so you can't add as you go.
# it takes a node and adds itself as the parents of whatever children it has. That way, the only node without a parent will be the root.
#(so every time you use kdtree, you need to also use this add parents function)
def add_parents(node):
    if node.left is not None:
        node.left.parent=node
    if node.right is not None:
        node.right.parent=node    

#traverses the tree and performs a function "callback" on each node it comes to.
def traverse_binary_tree(node, callback):
    if node is None:
        return
    traverse_binary_tree(node.left, callback)
    callback(node)
    traverse_binary_tree(node.right, callback)

#traverses the tree, and updates the sizes of each node by setting size_now equal to size_next and size_next equal to size_now.
def set_next_now_all(node):
    if node is None:
        return
    set_next_now_all(node.left)
    node.set_next_now()
    set_next_now_all(node.right)

#this function will add the proper number to size_next if the mut_key is already in the tree, and otherwise
#it will add the new node to the tree. ***This function specifically adds nodes to the global variable global_tree, which is why there's no tree input, only the node to add.
#it will update heights in the path of the added node,
#until it reaches the first issue, and then it will rotate and fix only those heights below the first issue, since the higher heights aren't affected.
# so with no issues, it will just update all the heights in the tree from parent to grandparent up to the root.
# new_node needs to have no parent to start with!
def add_to_tree(new_node):
    global global_tree
    current = global_tree
    done=0
    case=0
    XY_bigger=0
    update_needed=0
    while(not done):
        if current is None: #make sure this case only happens when the tree is empty.
            global_tree=Node(new_node.mut_key, new_node.s, new_node.size_now, new_node.size_next, new_node.left, new_node.right, None)
            done=1
        elif current.mut_key == new_node.mut_key:
            current.setSizeNext(current.size_next+new_node.size_next)
            done=1
        elif new_node.value < current.value:
            if current.left is not None:
                current=current.left
            else:
                current.left=new_node
                current.left.parent=current
                done=1
                update_needed=1
        else: # value >=current.value, so the new node falls to the right of the node you're currently on
            if current.right is not None:
                current=current.right
            else:
                current.right=new_node
                current.right.parent=current
                done=1
                update_needed=1
    if update_needed==1:
        difference=0
        while abs(difference)<=1:
            if current is None:
                return
            elif current.left is not None and current.right is not None:
                difference=current.left.height-current.right.height
                current.height=max(current.left.height, current.right.height)+1
                if abs(difference)<=1:
                    current=current.parent
            elif current.left is None and current.right is not None:
                difference=-current.right.height-1 #adding 1 here because we need to treat the empty child as having height -1, and making negative so that right heavy becomes negative.
                current.height=current.right.height+1
                if abs(difference)<=1:
                    current=current.parent
            else: #it had a left child but not a right child
                difference=current.left.height+1 #and adding 1 here for the same reason as above!
                current.height=current.left.height+1
                if abs(difference)<=1:
                    current=current.parent
            
        if difference < -1: #right heavy imbalances
            if current.right.right is not None and current.right.left is not None:
                if current.right.right.height>=current.right.left.height:
                    case=1
                    if current.right.right.height==current.right.left.height:
                        XY_bigger=1
            elif current.right.left is None:
                case=1
            else:
                case=2
            if case==1:
                Y=current.right
                B=current.right.left
                P=current.parent
                current.right=B
                if B is not None:
                    B.parent=current
                Y.left=current
                current.parent=Y
                Y.parent=P
                if P is not None and P.right==current:
                    P.right=Y
                if P is not None and P.left==current:
                    P.left=Y
                if P is None: #that means that current was the root, so we need to set a new root.
                    global_tree=current.parent
                if XY_bigger==1:
                    current.height=current.height-1
                    Y.height+=1
                else:
                    current.height=current.height-2
                father=P
                while father is not None:
                    if father.left is not None and father.right is not None:
                        father.height=max(father.left.height, father.right.height)+1
                    elif father.left is None:
                        father.height=father.right.height+1
                    else:
                        father.height=father.left.height+1
                    father=father.parent
            if case==2:
                Z=current.right
                Y=Z.left
                D=Z.right
                B=Y.left
                C=Y.right
                P=current.parent
                Y.left=current
                current.parent=Y
                Y.right=Z
                Z.parent=Y
                current.right=B
                if B is not None:
                    B.parent=current
                Z.left=C
                if C is not None:
                    C.parent=Z
                Z.right=D
                if D is not None:
                    D.parent=Z
                Y.parent=P
                if P is not None and P.right==current:
                    P.right=Y
                if P is not None and P.left==current:
                    P.left=Y
                if P is None: #that means that current was the root, so we need to set a new root.
                    global_tree=current.parent
                Z.height=Z.height-1
                Y.height+=1
                current.height=current.height-2
                father=P
                while father is not None:
                    if father.left is not None and father.right is not None:
                        father.height=max(father.left.height, father.right.height)+1
                    elif father.left is None:
                        father.height=father.right.height+1
                    else:
                        father.height=father.left.height+1
                    father=father.parent
            
        if difference > 1: #left heavy imbalances
            if current.left.left is not None and current.left.right is not None:
                if current.left.left.height>=current.left.right.height:
                    case=1
                    if current.left.left.height==current.left.right.height:
                        XY_bigger=1
            elif current.left.right is None:
                case=1
                #print("case 1")
            else:
                case=2
                #print("case 2")
            if case==1:
                Y=current.left
                B=current.left.right
                P=current.parent
                current.left=B
                if B is not None:
                    B.parent=current
                Y.right=current
                current.parent=Y
                Y.parent=P
                if P is not None and P.right==current:
                    P.right=Y
                if P is not None and P.left==current:
                    P.left=Y
                if P is None: #that means that current was the root, so we need to set a new root.
                    global_tree=current.parent
                if XY_bigger==1:
                    current.height=current.height-1
                    Y.height+=1
                else:
                    current.height=current.height-2
                father=P
                while father is not None:
                    if father.left is not None and father.right is not None:
                        father.height=max(father.left.height, father.right.height)+1
                    elif father.left is None:
                        father.height=father.right.height+1
                    else:
                        father.height=father.left.height+1
                    father=father.parent
            if case==2:
                Z=current.left
                Y=Z.right
                D=Z.left
                B=Y.right
                C=Y.left
                P=current.parent
                Y.right=current
                current.parent=Y
                Y.left=Z
                Z.parent=Y
                current.left=B
                if B is not None:
                    B.parent=current
                Z.right=C
                if C is not None:
                    C.parent=Z
                Z.left=D
                if D is not None:
                    D.parent=Z
                Y.parent=P
                if P is not None and P.right==current:
                    P.right=Y
                if P is not None and P.left==current:
                    P.left=Y
                if P is None: #that means that current was the root, so we need to set a new root.
                    global_tree=current.parent
                Z.height=Z.height-1
                Y.height+=1
                current.height=current.height-2
                father=P
                while father is not None:
                    if father.left is not None and father.right is not None:
                        father.height=max(father.left.height, father.right.height)+1
                    elif father.left is None:
                        father.height=father.right.height+1
                    else:
                        father.height=father.left.height+1
                    father=father.parent
                

    
#node is the old tree we are updating
# whole step 2 turns the global variable global_tree into the new tree representing the next generation. It traverses the tree and for every node, decides how many children
#with how many mutations each it has and then adds these children to the new tree. 
def whole_step_2(node):
    global global_tree
    next_gen=calc_z_d(node)
    if next_gen[0]!=0:
        new_node=Node(
                    node.mut_key, #has the same mut key because this is the set of children with no mutations.
                    node.s,
                    0,
                    next_gen[0],
                    None,
                    None,
                    None
                    )
        add_to_tree(new_node)
    for d in range(1,d_max+1):
        for i in range(next_gen[d]):
            positions=random.sample([j for j in range(r)],d) #chooses d positions from a set of positions without replacement
            directions=[0 for j in range(d)]
            for j in range(d):
                directions[j]=random.randrange(1,3) #chooses a mutation of either 1 or 2
            new_key=[0 for j in range(r)]
            for j in range(r):
                new_key[j]=node.mut_key[j]
            for j in range(d):
                new_key[positions[j]]+=directions[j]
                new_key[positions[j]]=new_key[positions[j]] % 3
            new_mut_key=[0 for j in range(r)]
            for j in range(r):
                new_mut_key[j]=new_key[j]
            new_node=Node(
                mut_key=new_mut_key,
                s=0,
                size_now=0,
                size_next=1,
                left=None,
                right=None,
                parent=None
                )
            add_to_tree(new_node)

#this just sets all the size_nexts to size_now in the global_tree created in step2. Then it returns this tree, which represents the next generation.
def step_3_and_4():
    global global_tree
    updated_tree=global_tree
    set_next_now_all(updated_tree)
    return updated_tree

def increment_pop_size(node):
    global pop_size
    pop_size+=node.size_now

#with the help of increment_pop_size, this function counts the population size of the tree.
def count_pop_size(tree):
    global pop_size
    pop_size=0
    traverse_binary_tree(tree,increment_pop_size)

#this function draws viruses determined by the bottleneck size and the random vector of numbers generated in the 'bottleneck' function, and puts the in the new tree.
def get_new_pop(node, viruses):
    global counter
    global counter2
    global bottleneck_size
    global global_tree
    global freq_count_mat
    if node is None:
        return
    get_new_pop(node.left, viruses)
    #the actual thing you do at each node
    for j in range(node.size_now):
        counter+=1
        freq_count_mat.matrix[counter]=node.mut_key
        if counter2<bottleneck_size: #if it's bigger, then we've pulled all the viruses we need to for the next generation, and can skip this step.
            if counter==viruses[counter2]: #that means this is the next virus in the list that we need to pull for the next gen. Have to update counter2 and add node to the new tree.
                counter2+=1
                new_node=Node(
                    mut_key=node.mut_key,
                    s=node.s,
                    size_now=0,
                    size_next=1,
                    left=None,
                    right=None,
                    parent=None
                    )
                add_to_tree(new_node)
    get_new_pop(node.right, viruses)  

#this function makes a new_tree which is the bottlenecked population. This is done at the ned of every passage.
def bottleneck(tree):
    global global_tree
    global pop_size
    global freq_count_mat
    global counter
    global counter2
    global_tree=None
    counter=-1 #have to set counters back to their orignal values immediately before traversing the tree in get_new_pop to be sure that they are at -1 and 0 when you begin to use them.
    counter2=0
    freq_count_mat.set_row_num(pop_size)
    count_pop_size(tree)
    viruses=random.sample([i for i in range(pop_size)],bottleneck_size)
    viruses.sort()
    get_new_pop(tree, viruses)
    new_tree=Node([1],1,1,1,None,None,None)
    new_tree.left=global_tree.left
    new_tree.right=global_tree.right
    new_tree.mut_key=global_tree.mut_key
    new_tree.s=global_tree.s
    new_tree.size_now=global_tree.size_now
    new_tree.size_next=global_tree.size_next
    new_tree.left.parent=new_tree
    new_tree.right.parent=new_tree
    return new_tree

#this function gets all the frequencies of mutations at the end of a passage before the bottleneck happens.
def make_freq_vector(freq_count_mat):
    freq_vector=[]
    for i in range(len(freq_count_mat.matrix[0])):
        count_0s=0
        count_2s=0
        for j in range(pop_size):
            if freq_count_mat.matrix[j][i]==0:
                count_0s+=1
            if freq_count_mat.matrix[j][i]==2:
                count_2s+=1
        freq_0s=count_0s/pop_size
        freq_2s=count_2s/pop_size
        if freq_0s!=0:
            freq_vector.append(freq_0s)
        if freq_2s!=0:
            freq_vector.append(freq_2s)
    return freq_vector

#this is the actual code, first making the P0 population (which for now is just a clonal population of 200 viruses), and then running the simulation. The output is freq_vector, but
#for now I don't print it because it's very large.
s_list=[0]
size_now_list=[200]
size_next_list=[0]
mut_list=[[1 for i in range(13000)]]

#initializing the tree with the the P0 population information
tree = kdtree(mut_list, s_list, size_now_list, size_next_list)
traverse_binary_tree(tree,add_parents) #right now this does nothing because there's only one node, but it will become necessary when using an actual P0 approximation for the starting population.

#variables to change
passages=1
generations=8

for j in range(passages):
    for i in range(generations):
        traverse_binary_tree(tree,whole_step_2)
        tree=step_3_and_4()
        global_tree=None
        print("generation")
        print(i+1)
        print("population size")
        count_pop_size(tree)
        print(pop_size)
        print("time is:")
        print(datetime.datetime.time(datetime.datetime.now()))
        print("memory use for tree is:")
        print(sys.getsizeof(tree))
    tree=bottleneck(tree)
    set_next_now_all(tree)
    global_tree=None
    make_freq_vector(freq_count_mat)
    print("freq_vector done, code done.")
    freq_count_mat.set_row_num(1)
