class Paper:
  def __init__(self, title, abstract, text=None):
    self.title = title
    self.abstract = abstract
    self.text = text
  
  def __eq__(self, other):
    return (self.title.lower() == other.title.lower()) and (self.abstract.lower() == other.abstract.lower())

  def __hash__(self):
    return hash(self.title + self.abstract)

  def __str__(self):
    return self.title