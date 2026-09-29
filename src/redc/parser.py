from lark import Lark

parser = Lark.open("src/grammar/redc.lark", parser="lalr")

with open("examples/uint8_add.rc") as file:
    tree = parser.parse(file.read())

print(tree.pretty())