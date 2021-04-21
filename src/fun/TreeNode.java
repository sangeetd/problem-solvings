/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

/**
 *
 * @author RAVI
 */
public class TreeNode<T> {
    private T data;
    private TreeNode<T> left;
    private TreeNode<T> right;
    private TreeNode<T> random;

    public TreeNode(T data) {
        this.data = data;
        this.left = null;
        this.right = null;
    }
    
    public TreeNode(T data, TreeNode<T> left, TreeNode<T> right) {
        this.data = data;
        this.left = left;
        this.right = right;
    }

    protected T getData() {
        return data;
    }

    protected void setData(T data) {
        this.data = data;
    }

    protected TreeNode<T> getLeft() {
        return left;
    }

    protected void setLeft(TreeNode<T> left) {
        this.left = left;
    }

    protected TreeNode<T> getRight() {
        return right;
    }

    protected void setRight(TreeNode<T> right) {
        this.right = right;
    }

    public TreeNode<T> getRandom() {
        return random;
    }

    public void setRandom(TreeNode<T> random) {
        this.random = random;
    }

    @Override
    public String toString() {
        return data.toString();
    }
    
}
