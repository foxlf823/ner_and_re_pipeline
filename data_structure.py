
class Entity:
    def __init__(self):
        self.id = None
        self.type = None
        self.start = None
        self.end = None
        self.text = None
        self.sent_idx = None
        self.tf_start = None
        self.tf_end = None

    def create(self, id, type, start, end, text, sent_idx, tf_start, tf_end):
        self.id = id
        self.type = type
        self.start = start
        self.end = end
        self.text = text
        self.sent_idx = sent_idx
        self.tf_start = tf_start
        self.tf_end = tf_end

    def append(self, start, end, text, tf_end):

        whitespacetoAdd = start - self.end
        for _ in range(whitespacetoAdd):
            self.text += " "
        self.text += text

        self.end = end
        self.tf_end = tf_end

    def getlength(self):
        return self.end-self.start

    def equals(self, other):
        if self.type == other.type and self.start == other.start and self.end == other.end:
            return True
        else:
            return False


class Relation:
    def __init__(self):
        self.id = None
        self.node1 = None
        self.node2 = None
        self.type = type

    def create(self, id, type, node1, node2):
        self.id = id
        self.type = type
        self.node1 = node1
        self.node2 = node2

    def equals(self, other):
        if self.type == other.type and self.node1.equals(other.node1) and self.node2.equals(other.node2):
            return True
        elif self.type == other.type and self.node1.equals(other.node2) and self.node2.equals(other.node1):
            return True
        else:
            return False

