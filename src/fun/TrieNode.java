/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fun;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author RAVI
 */
public class TrieNode {
    
    private Map<Character, TrieNode> node;
    private long wordCounterAtEndOfWord=0;
    private boolean endOfWord;

    public TrieNode() {
        this.node = new HashMap<>();
    }

    public TrieNode(Character c) {
        this();
        node.put(c, new TrieNode());
    }
    
    public TrieNode(Character c, TrieNode n) {
        this();
        node.put(c, n);
    }

    public Map<Character, TrieNode> charNode() {
        return node;
    }

    public void setCharMap(Map<Character, TrieNode> node) {
        this.node = node;
    }
    
    public boolean isEndOfWord() {
        return endOfWord;
    }

    public void setEndOfWord(boolean endOfWord) {
        this.endOfWord = endOfWord;
    }

    public long getWordCount() {
        return wordCounterAtEndOfWord;
    }

    public void increaseWordCount() {
        this.wordCounterAtEndOfWord++;
    }
    
    public void decreaseWordCount() {
        this.wordCounterAtEndOfWord--;
    }
    
    
    
}
