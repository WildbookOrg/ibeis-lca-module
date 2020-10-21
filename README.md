# WBIA LCA Plugin
An example of how to design and use a Python module as a plugin in the WBIA system

# Installation

Install this plugin as a Python module using

```bash
cd ~/code/wbia_lca/
python setup.py develop
```

# REST API

With the plugin installed, register the module name with the `IBEISControl.py` file
in the wbia repository located at `wbia/wbia/control/IBEISControl.py`.  Register
the module by adding the string (for example, `wbia_lca`) to the
list `AUTOLOAD_PLUGIN_MODNAMES`.

Then, load the web-based WBIA service and open the URL that is registered with
the `@register_api decorator`.

```bash
cd ~/code/wbia/
python dev.py --web
```

Navigate in a browser to http://127.0.0.1:5000/api/plugin/example/helloworld/ where
this returns a formatted JSON response, including the serialized returned value
from the `wbia_lca_hello_world()` function

```
{"status": {"cache": -1, "message": "", "code": 200, "success": true}, "response": "[wbia_lca] hello world with IBEIS controller <IBEISController(testdb1) at 0x11e776e90>"}
```

# Python API

```bash
python

Python 2.7.14 (default, Sep 27 2017, 12:15:00)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.37)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import wbia
>>> ibs = wbia.opendb()

[ibs.__init__] new IBEISController
[ibs._init_dirs] ibs.dbdir = u'/Datasets/testdb1'
[depc] Initialize ANNOTATIONS depcache in u'/Datasets/testdb1/_ibsdb/_wbia_cache'
[depc] Initialize IMAGES depcache in u'/Datasets/testdb1/_ibsdb/_wbia_cache'
[ibs.__init__] END new IBEISController

>>> ibs.wbia_lca_hello_world()
'[wbia_lca] hello world with IBEIS controller <IBEISController(testdb1) at 0x10b24c9d0>'
```

The function from the plugin is automatically added as a method to the ibs object
as `ibs.wbia_lca_hello_world()`, which is registered using the
`@register_ibs_method decorator`.
