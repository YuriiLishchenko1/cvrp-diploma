import math, json
def distance(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
def tour_length(route, coords):
    return sum(distance(coords[i], coords[j])
               for i,j in zip(route, route[1:]))
def write_json(obj, name="result.json"):
    with open(name,"w",encoding="utf8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)
