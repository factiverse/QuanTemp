"""Quantity segmenter that returns the quantitiative segments within a claim

    Returns:
        List[str]: List of quantitative segments
    """
from pycorenlp import StanfordCoreNLP
from typing import Any, List, Tuple

from collections import deque


class Node():
    """encodes node of the constituency parse tree
    """

    def __init__(self, value=None) -> None:
        self.value = value
        self.children = []

    def __eq__(self, other: Any) -> bool:
        """ Check equality of node values
        Overrides the default implementation
        For our purposes, at the node level, we consider a node equal
        to another if they share the same value
        and their children share the same value"""
        if isinstance(other, Node):
            is_equal = self.value == other.value and len(
                self.children) == len(other.children)
            if is_equal:
                for child_ind in range(len(self.children)):
                    if self.children[child_ind].value != other.children[child_ind].value:
                        return False
                return is_equal
        return False

    def __ne__(self, other: Any) -> bool:
        """Check equality of node values

        Args:
            other (Any): Node value

        Returns:
            bool: indication of whether nodes are equal
        """
        return not self.__eq__(other)

    def set_value(self, value):
        self.value = value

    def set_child_at_index(self, child, index=-1):
        if index == -1:
            self.children.append(child)
        else:
            self.children[index] = child

    def get_value(self):
        return self.value

    def get_child_at_index(self, index=-1):
        assert abs(index) < len(self.children)
        return self.children[index]

    def get_children(self):
        return self.children


class Tree():
    def __init__(self, root):
        '''
        Initialize tree with root
        @param root: root is an object of type Node, representing root 
        of the binary tree
        '''
        self.root = root

    def __eq__(self, other):
        """Overrides the default implementation
        For our purposes, at the node level, we consider a node equal 
        to another if they share the same value
        and their children share the same value"""
        if isinstance(other, Tree):
            return self.get_prefix_traversal(
                self.root) == self.get_prefix_traversal(other.root)
        return False

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)

    def get_prefix_traversal(self, start_node=None):
        '''
        Get prefix traversal of tree
        :param start_node: node in the tree, which is root of the subtree we
        would like a prefix traversal for
        :return: prefix traversal
        '''

        if start_node == None:
            return []

        def is_pos_tag(node):
            children = node.get_children()
            return len(children) == 1 and children[0] and len(children[0].get_children()) == 0

        # If we reach a POS tag we follow the same format as
        # the Stanford parser prefix
        if is_pos_tag(start_node):
            prefix_traversal = [
                "(", start_node.value, start_node.get_children()[0].value, ")"]
            return prefix_traversal

        prefix_traversal = ["(", start_node.value]

        for each_child in start_node.get_children():
            prefix_traversal += self.get_prefix_traversal(each_child)

        prefix_traversal.append(")")
        return prefix_traversal

    def get_ancestors(self, query_node, current_node, path=[]):
        '''
        Get all ancestors along the path from root to a query node
        :param query_node: Node object whose ancestors we want
        '''

        if current_node == None:
            return []

        # If we reach our query node
        if current_node == query_node:
            return [current_node]

        found_query = False
        for each_child in current_node.get_children():
            path_to_leaf = self.get_ancestors(query_node, each_child, path)

            if len(path_to_leaf) > 0:
                found_query = True
                path = path_to_leaf
                path.append(current_node)
                return path

        if not found_query:
            return []

    def get_nodes_with_value(self, query_value, start_node, path=[]):
        '''
        Get all nodes with a particular value
        :param query_node: Node object whose ancestors we want
        '''
        nodes_with_value = []

        def visit_all_nodes(query_value, current_node):

            if current_node == None:
                return

            # If we reach our query node
            if current_node.value == query_value:
                nodes_with_value.append(current_node)

            for each_child in current_node.get_children():
                visit_all_nodes(query_value, each_child)

        visit_all_nodes(query_value, start_node)
        return nodes_with_value


def get_subtree(parts, start_index):
    '''
    get_subtree returns the end_index in the parse prefix expression
    corresponding to a given subtree
    :param parts: prefix expression of constituency parse of the sentence
    in the form of a list
    :param start_index: start of the subtree
    :return: end_index: end of the subtree in the prefix expression
    '''

    # This is invalid
    if parts[start_index] != '(':
        return -1

    # Create a deque to use it as a stack.
    stack = deque()

    for element_index in range(start_index, len(parts)):

        # Pop a starting bracket
        # for every closing bracket
        if parts[element_index] == ')':
            stack.popleft()

        # Push all starting brackets
        elif parts[element_index] == '(':
            stack.append(parts[element_index])

        # If stack becomes empty
        if not stack:
            end_index = element_index
            return end_index

    return -1


def get_tree(parts: List[Any], start_index: int) -> Node:
    """returns tree structure

    Args:
        parts (Node): subtree
        start_index (_type_): start index of root node

    Returns:
        Node: returns tree structure
    """

    if start_index >= len(parts):
        return Node(None)

    # Current Node in prefix expression becomes value of the node
    value_index = start_index + 1  # TAG after opening brace
    value = parts[value_index]
    node = Node(value)

    subtree_start = value_index + 1
    while parts[subtree_start] == "(":
        subtree_end = get_subtree(parts, subtree_start)
        if subtree_end - subtree_start == 3:
            pos_tag = parts[subtree_start + 1]
            word = parts[subtree_end - 1]

            child_node = Node(pos_tag)
            word_node = Node(word)

            child_node.set_child_at_index(word_node, -1)
            node.set_child_at_index(child_node, -1)
        else:
            node.set_child_at_index(get_tree(parts, subtree_start), -1)
        subtree_start = subtree_end + 1

    return node


def construct_parse(parse: str) -> Tree:
    """Converts prefix expression of parse to an expression tree

    Args:
        parse (str): constituency parse of the sentence in prefix

    Returns:
        Tree: expression tree of constituency parse of the sentence
    """
    parse = parse.replace("(", " ( ")
    parse = parse.replace(")", " ) ")
    parts = parse.split()
    start_index = 0

    root = get_tree(parts, start_index)
    parse_tree = Tree(root)

    return parse_tree


class Segmenter():
    """
    Inputs:
        args:
            sentence (str) : sentence to extract quantity mentions from
    Outputs:
        quantity mentions (list[str]) : list of noun phrases
        containing quantities from parse
    """

    def __init__(self):
        # pass
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    def segment(self, text) -> Tuple[List[List[str]], List[Node]]:
        '''Extracts and returns all quantity mentions in a text.
        Quantity mentions are defined as least ancestor noun phrases
        containing quantities.
        Args:
        text (str): Sentence (typically) that we want to extract 
        quantity mentions from
        Returns:
          quantity_mentions, noun_phrases: List of quantity mentions 
          found in sentence'''

        noun_phrases, quantity_phrases = [], []
        parse = self.nlp.annotate(text, properties={
            'annotators': 'parse',
            'outputFormat': 'json',
            'timeout': 10000,
        })
        # print(parse)
        for each_parse_sentence in [parse['sentences'][0]["parse"]]:
            parse_tree = construct_parse(each_parse_sentence)
            quantities = parse_tree.get_nodes_with_value("CD", parse_tree.root)
            print("quantities",quantities[1].children[0].value)
            ancestor_chains = [parse_tree.get_ancestors(
                quantity, parse_tree.root) for quantity in quantities]

            for each_chain in ancestor_chains:
                chain_values = [node.value for node in each_chain]
                lca_index = -1

                # Finds least common noun phrase
                if "NP" in chain_values:
                    lca_index = chain_values.index("NP")

                # FInds least common NP-TMP phrase
                if "NP-TMP" in chain_values:
                    tmp_index = chain_values.index("NP-TMP")
                    if lca_index == -1 or tmp_index < lca_index:
                        lca_index = tmp_index

                # Extracts first noun phrase node in each ancestor chain
                if lca_index != -1:
                    noun_phrases.append(each_chain[lca_index])

            quantity_phrases += [parse_tree.get_prefix_traversal(
                noun_node) for noun_node in noun_phrases]

        return quantity_phrases, noun_phrases


if __name__ == "__main__":
    """Example quantity segment identification
    """
    segmenter = Segmenter()
    quantity_phrases, noun_phrases = segmenter.segment("""During his 29
      seasons as chairman of the chicago bulls ,
      the team captured the title seven times ,
      including in 2000""")
    print("quantity_phrases", quantity_phrases)
