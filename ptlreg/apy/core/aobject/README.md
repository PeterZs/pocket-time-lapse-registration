


# Apy Objects

### [Serializable](Serializable.py)
provides parent/base implementations of:
- `get_info` 
− `set_info`
- `to_dictionary`
- `init_from_dictionary`

### [Sequencable](Sequencable.py)
Mixin for things to be sequencable. Defines:
- `_next`
- `_prev`
- `_list`

Not sure this is used right now in stripped down version

### [HasTags](HasTags.py)
Mixin that adds `_tags`, which get serialized appropriately. Tags is a dictionary that by default just maps each tag to `True` or `False`


### [AObject](AObject.py)
The main base class for most objects in the apy library. It provides:




