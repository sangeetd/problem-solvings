/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import javafx.util.Pair;

/**
 *
 * @author RAVI
 */
public class BinaryTree<T> {

    private TreeNode<T> root;
    private int height = 0;

    public BinaryTree() {
        this.root = null;
    }

    public BinaryTree(TreeNode rootNode) {
        root = rootNode;
    }

    public BinaryTree(T data) {
        root = new TreeNode(data, null, null);
    }

    public BinaryTree(T data, TreeNode left, TreeNode right) {
        root = new TreeNode(data, left, right);
    }

    public TreeNode<T> getRoot() {
        return root;
    }

    public void setRoot(TreeNode<T> root) {
        this.root = root;
    }

    public void insert(T data) {

        if (root == null) {
            this.root = new TreeNode(data);
            return;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode t = q.poll();
            if (t.getLeft() == null) {
                t.setLeft(new TreeNode(data));
                return;
            } else if (t.getRight() == null) {
                t.setRight(new TreeNode(data));
                return;
            }

            q.add(t.getLeft());
            q.add(t.getRight());

        }

    }

    public void treeBFS() {

        //tree bfs is also level order
        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }

        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);

        while (!q.isEmpty()) {

            TreeNode t = q.poll();
            System.out.print(t.toString() + " ");
            if (t.getLeft() != null) {
                q.add(t.getLeft());
            }

            if (t.getRight() != null) {
                q.add(t.getRight());
            }

        }

    }

    private void inorder(TreeNode node) {

        if (node == null) {
            return;
        }

        if (node.getLeft() != null) {
            inorder(node.getLeft());
        }

        System.out.print(node.toString() + " ");

        if (node.getRight() != null) {
            inorder(node.getRight());
        }

    }

    public void treeInorder() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            inorder(this.root);
        }

    }

    private void preorder(TreeNode node) {

        if (node == null) {
            return;
        }

        System.out.print(node.toString() + " ");

        if (node.getLeft() != null) {
            preorder(node.getLeft());
        }

        if (node.getRight() != null) {
            preorder(node.getRight());
        }

    }

    public void treePreorder() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            preorder(this.root);
        }

    }

    private void postorder(TreeNode node) {

        if (node == null) {
            return;
        }

        if (node.getLeft() != null) {
            postorder(node.getLeft());
        }

        if (node.getRight() != null) {
            postorder(node.getRight());
        }

        System.out.print(node.toString() + " ");

    }

    public void treePostorder() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            postorder(this.root);
        }

    }

    private void leftOuterBoundary_TopToBottom(TreeNode node, Set<T> set) {

        if (node == null) {
            return;
        }

        //System.out.print(node.toString()+" ");
        set.add((T) node.getData());
        if (node.getLeft() != null) {
            leftOuterBoundary_TopToBottom(node.getLeft(), set);
        }

    }

    private void rightOuterBoundary_BottomToTop(TreeNode node, Set<T> set) {

        if (node == null) {
            return;
        }

        //going down deep in right tree
        //print last of that root and recursive back up to top
        if (node.getRight() != null) {
            rightOuterBoundary_BottomToTop(node.getRight(), set);
        }
        //System.out.print(node.toString()+" ");
        set.add((T) node.getData());
    }

    private void leafLevelBoundary_LeftToRight(TreeNode node, Set<T> set) {

        if (node == null) {
            return;
        }

        if (node.getLeft() != null) {
            leafLevelBoundary_LeftToRight(node.getLeft(), set);
        }

        if (node.getLeft() == null && node.getRight() == null) {
            set.add((T) node.getData());
            //System.out.print(node.toString()+" ");
        }

        if (node.getRight() != null) {
            leafLevelBoundary_LeftToRight(node.getRight(), set);
        }

    }

    private void outerBoundary(Set<T> set) {

        leftOuterBoundary_TopToBottom(root, set);

        leafLevelBoundary_LeftToRight(root.getLeft(), set);
        leafLevelBoundary_LeftToRight(root.getRight(), set);

        rightOuterBoundary_BottomToTop(root, set);

    }

    public void treeOuterBoundry() {

        //from root to left then to leaf and back to root
        //in anticlockwise fasion
        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            Set<T> set = new LinkedHashSet<>();
            outerBoundary(set);
            System.out.print(set.toString());
        }

    }

    private void zigzag(boolean side) {

        Stack<TreeNode> startStack = new Stack<>();
        Stack<TreeNode> internalStack = new Stack<>();

        startStack.push(root);
        boolean leftToRight = side;
        while (!startStack.isEmpty()) {

            TreeNode t = startStack.pop();

            System.out.print(t.toString() + " ");

            if (leftToRight) {

                if (t.getRight() != null) {
                    internalStack.push(t.getRight());
                }

                if (t.getLeft() != null) {
                    internalStack.push(t.getLeft());
                }

            } else {

                if (t.getLeft() != null) {
                    internalStack.push(t.getLeft());
                }

                if (t.getRight() != null) {
                    internalStack.push(t.getRight());
                }

            }

            if (startStack.isEmpty()) {

                leftToRight = !leftToRight;
                Stack<TreeNode> temp = startStack;
                startStack = internalStack;
                internalStack = temp;

            }

        }

    }

    public void treeZigZag(boolean leftToRight) {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        } else {
            zigzag(leftToRight);
        }

    }

    private void kSumPathAndDeleteOtherNode(TreeNode node, int k, int sum, TreeNode parent) {

        sum += (int) node.getData();

        if (node.getLeft() != null) {
            kSumPathAndDeleteOtherNode(node.getLeft(), k, sum, node);
        }

        //leaf nodes
        //System.out.println("|| sum: " + sum + " node " + (int) node.getData());
        if (sum >= k) {
            //to check other nodes also adjust sum here only
            sum -= (int) node.getData();
            //System.out.println("-- sum: " + sum + " node " + (int) node.getData());
            return;
        } else if (sum < k && (node.getLeft() == null && node.getRight() == null)) {
            //to check other nodes also adjust sum here only
            sum -= (int) node.getData();
            //System.out.println(".. sum: " + sum + " node " + (int) node.getData() + " parent " + (int) parent.getData());
            if (node == parent.getLeft()) {
                parent.setLeft(null);
                //System.out.println("hhh ");
            } else if (node == parent.getRight()) {
                parent.setRight(null);
                //System.out.println("ggg");
            }

            return;
        }

        if (node.getRight() != null) {
            kSumPathAndDeleteOtherNode(node.getRight(), k, sum, node);
        }

    }

    public void kSumPathOfTree(int k) throws Exception {

        if (getRoot() == null) {
            System.out.println("Tree is empty");
            return;
        }

        if (!(getRoot().getData() instanceof Integer)) {
            throw new Exception("K sum path can be implemented as Integer generic type");
        }

        kSumPathAndDeleteOtherNode(root, k, 0, null);

    }

    public int treeHeightWithNode(TreeNode node) {

        if (node == null) {
            return 0;
        }

        return Math.max(treeHeightWithNode(node.getLeft()), treeHeightWithNode(node.getRight())) + 1;

    }

    public int treeHeight() {
        if (getRoot() == null) {
            System.out.println("Tree is empty");
            return 0;
        }

        return treeHeightWithNode(getRoot());

    }
    
    private TreeNode<T> buildTreeFromInorderPreorder_Helper(int preIndex, int inStart, int inEnd, T[] inorder, T[] preorder){
        
        if(preIndex >= preorder.length || inStart > inEnd){
            return null;
        }
        
        TreeNode<T> root = new TreeNode<>(preorder[preIndex]);
        
        int index = inStart;
        for(; index<=inEnd; index++){
            if(preorder[preIndex] == inorder[index]){
                break;
            }
        }
        
        root.setLeft(buildTreeFromInorderPreorder_Helper(preIndex+1, inStart, index - 1, inorder, preorder));
        root.setRight(buildTreeFromInorderPreorder_Helper(preIndex+1+index-inStart, index + 1, inEnd, inorder, preorder));
        
        return root;
    }
    
    public BinaryTree<T> buildTreeFromInorderPreorder(T[] inorder, T[] preorder) {

        if (inorder.length != preorder.length) {
            throw new RuntimeException("Tree nodes are not equal");
        }
        TreeNode<T> root = buildTreeFromInorderPreorder_Helper(0, 0, inorder.length-1, preorder, inorder);
        return new BinaryTree<T>(root);
    }
    

    public void treeTopView() {
        //inner class scope limited to this method only
        class NodePair {

            TreeNode node;
            int horizontalDistance;

            public NodePair(TreeNode node, int horizontalDistance) {
                this.node = node;
                this.horizontalDistance = horizontalDistance;
            }

        }

        Queue<NodePair> q = new LinkedList<>();
        Map<Integer, TreeNode> map = new TreeMap<>();

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }

        q.add(new NodePair(root, 0));

        while (!q.isEmpty()) {

            NodePair t = q.poll();

            if (!map.containsKey(t.horizontalDistance)) {
                map.put(t.horizontalDistance, t.node);
            }

            if (t.node.getLeft() != null) {
                q.add(new NodePair(t.node.getLeft(), t.horizontalDistance - 1));
            }

            if (t.node.getRight() != null) {
                q.add(new NodePair(t.node.getRight(), t.horizontalDistance + 1));
            }

        }

        for (Map.Entry<Integer, TreeNode> e : map.entrySet()) {
            System.out.print(e.getValue().getData() + " ");
        }

    }

    private void leftViewHelper(TreeNode<T> root, int level, TreeMap<Integer, T> result) {

        if (root == null) {
            return;
        }

        if (!result.containsKey(level)) {
            result.put(level, root.data);
        }

        leftViewHelper(root.getLeft(), level + 1, result);
        leftViewHelper(root.getRight(), level + 1, result);

    }

    public List<T> leftView() {
        // Your code here
        Map<Integer, T> result = new TreeMap<>();
        List<T> output = new ArrayList<>();
        if (root == null) {
            return output;
        }

        leftViewHelper(this.root, 0, (TreeMap<Integer, T>) result);

        result.entrySet().stream().forEach((e) -> {
            output.add(e.getValue());
        });

        return output;

    }
    
    private void rightViewHelper(TreeNode<T> root, int level, TreeMap<Integer, T> result) {

        if (root == null) {
            return;
        }

        if (!result.containsKey(level)) {
            result.put(level, root.data);
        }

        rightViewHelper(root.getRight(), level + 1, result);
        rightViewHelper(root.getLeft(), level + 1, result);
        
    }

    public List<T> rightView() {
        // Your code here
        Map<Integer, T> result = new TreeMap<>();
        List<T> output = new ArrayList<>();
        if (root == null) {
            return output;
        }

        rightViewHelper(this.root, 0, (TreeMap<Integer, T>) result);

        result.entrySet().stream().forEach((e) -> {
            output.add(e.getValue());
        });

        return output;

    }

    public void treeInorderIterative() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }

        Stack<Pair<TreeNode<T>, Integer>> st = new Stack<>();
        st.push(new Pair<>(getRoot(), 0));

        while (!st.isEmpty()) {

            Pair<TreeNode<T>, Integer> p = st.pop();
            TreeNode<T> t = p.getKey();
            Integer status = p.getValue();

            if (t == null || status == 3) {
                continue;
            }

            st.push(new Pair<>(t, status + 1));

            if (status == 0) {
                st.push(new Pair<>(t.getLeft(), 0));
            }
            if (status == 1) {
                System.out.print(t.getData() + " ");
            }
            if (status == 2) {
                st.push(new Pair<>(t.getRight(), 0));
            }

        }

    }

    public void treePreorderIterative() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }

        Stack<Pair<TreeNode<T>, Integer>> st = new Stack<>();
        st.push(new Pair<>(getRoot(), 0));

        while (!st.isEmpty()) {

            Pair<TreeNode<T>, Integer> p = st.pop();
            TreeNode<T> t = p.getKey();
            Integer status = p.getValue();

            if (t == null || status == 3) {
                continue;
            }

            st.push(new Pair<>(t, status + 1));

            if (status == 0) {
                System.out.print(t.getData() + " ");
            }
            if (status == 1) {
                st.push(new Pair<>(t.getLeft(), 0));
            }
            if (status == 2) {
                st.push(new Pair<>(t.getRight(), 0));
            }

        }

    }

    public void treePostorderIterative() {

        if (root == null) {
            System.out.println("Tree is empty");
            return;
        }

        Stack<Pair<TreeNode<T>, Integer>> st = new Stack<>();
        st.push(new Pair<>(getRoot(), 0));

        while (!st.isEmpty()) {

            Pair<TreeNode<T>, Integer> p = st.pop();
            TreeNode<T> t = p.getKey();
            Integer status = p.getValue();

            if (t == null || status == 3) {
                continue;
            }

            st.push(new Pair<>(t, status + 1));

            if (status == 0) {
                st.push(new Pair<>(t.getLeft(), 0));
            }
            if (status == 1) {
                st.push(new Pair<>(t.getRight(), 0));
            }
            if (status == 2) {
                System.out.print(t.getData() + " ");
            }

        }

    }

}
