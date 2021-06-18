class Animal:
  def __init__(self, animalName):
    self.animalName = animalName


  def pr(self):
    print(self.animalName, 'is an animal.');

# Mammal inherits Animal
class Mammal(Animal):
  def __init__(self, mammalName):
    self.mammalName = mammalName
    super().__init__(mammalName)

  def pr(self):
    print(self.mammalName, 'is a mammal.')
    super().pr()

# CannotFly inherits Mammal
class CannotFly(Mammal):
  def __init__(self, mammalThatCantFly):
    self.mammalThatCantFly = mammalThatCantFly
    super().__init__(mammalThatCantFly)

  def pr(self):
    print(self.mammalThatCantFly, "cannot fly.")
    super().pr()


def main():
  cantfly = CannotFly("Fish")
  print('hi')
  cantfly.pr()

if __name__ == '__main__':
  main()