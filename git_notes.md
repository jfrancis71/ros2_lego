# Git Notes

To create a repo:
```
git init
```

To add a remote reference:
```
git remote add remote_lego https://github.com/jfrancis71/ros2_lego.git
```
This simply adds remote_lego as a reference for the above https address. It does not download anything.

To download a remote branch:
```
git fetch remote_lego remote_branch_name
```

To Fetch remote branch and create a local tracking branch:
```
git fetch remote_lego remote_branch_name:local_branch_name
```


```
git cat-file -p hashid
```
Prints out contents of object, -p option says figure out type, eg tree, commit, blob etc

```
git branch --all
```
shows all branches both local and remote

To show which branches are tracking which remotes:
```
git branch -vv
```
Also:
```
git remote show remote_lego
```

To show all objects:
```
git cat-file --batch-all-objects --batch-check
```
