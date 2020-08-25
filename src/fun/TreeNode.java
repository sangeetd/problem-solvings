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
    T data;
    TreeNode left;
    TreeNode right;

    public TreeNode(T data) {
        this.data = data;
        this.left = null;
        this.right = null;
    }
    
    public TreeNode(T data, TreeNode left, TreeNode right) {
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

    protected TreeNode getLeft() {
        return left;
    }

    protected void setLeft(TreeNode left) {
        this.left = left;
    }

    protected TreeNode getRight() {
        return right;
    }

    protected void setRight(TreeNode right) {
        this.right = right;
    }

    @Override
    public String toString() {
        return data.toString();
    }
    
    
    
}
