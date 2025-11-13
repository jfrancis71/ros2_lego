# Git Notes

To create a repo:
```
git init
```

To add a remote reference:
```
git remote add remote_lego https://github.com/jfrancis71/ros2_lego.git
```

To Fetch remote branch into local branch:
```
git fetch remote_lego remote_branch_name:local_branch_name
```

```
git cat-file -p hashid
```
Prints out contents of object, -p option says figure out type, eg tree, commit, blob etc
