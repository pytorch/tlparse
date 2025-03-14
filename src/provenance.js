let preGradGraphData = null;
let postGradGraphData = null;
let codeData = null;
let cppCodeData = null;

let preToPost = {};
let postToPre = {};
let pyCodeToPost = {};
let postToPyCode = {};
let postToCppCode = {};
let cppCodeToPost = {};

let jsonData = null;

/**
 * Converts node-based mappings to line number-based mappings for visualization.
 * 
 * This function processes four types of files and their relationships:
 * 1. Pre-grad graph (FX IR before autograd and any pre_grad pass)
 * 2. Post-grad graph (FX IR after autograd)
 * 3. Generated Python triton code (produced by JIT inductor)
 * 3. Generated C++ code  (produced by AOT inductor)
 * 
 * The conversion happens in several steps:
 * 
 * 1. First, it creates lookup maps that associate node names with line numbers:
 *    - For pre/post grad graphs: Extracts node names from lines like "node_name = ..." or "node_name: ... = ..."
 *    - For C++/python code: Identifies kernel definitions and their associated line ranges
 * 
 * 2. Then, it processes four types of mappings:
 *    - preToPost: Maps pre-grad nodes to post-grad nodes
 *    - postToPre: Maps post-grad nodes back to pre-grad nodes
 *    - cppCodeToPost: Maps C++/triton kernel lines to post-grad nodes
 *    - postToCppCode: Maps post-grad nodes to C++ kernel lines
 * 
 * 3. For each mapping type, it:
 *    - Looks up the line numbers for the source nodes
 *    - Looks up the line numbers for the target nodes
 *    - Creates a new mapping using line numbers instead of node names
 * 
 * Special handling for C++ code:
 * - C++ kernels span multiple lines (from kernel definition to launch)
 * - Each kernel's line range includes:
 *   * The nullptr check line
 *   * All lines up to and including the launchKernel call
 *   * One line after the launch for completeness
 * 
 * The function updates these global variables:
 * - preToPost: {sourceLineNum: [targetLineNums]}
 * - postToPre: {sourceLineNum: [targetLineNums]}
 * - cppCodeToPost: {sourceLineNum: [targetLineNums]}
 * - postToCppCode: {sourceLineNum: [targetLineNums]}
 * 
 * These mappings enable the UI to highlight corresponding lines
 * across different views when a user clicks on a line.
 */
function convertNodeMappingsToLineNumbers() {
    if (!nodeMappings) {
        console.warn('No node mappings available');
        return;
    }

    function validLine(line) {
        const stripped = line.trim();
        return stripped && !stripped.startsWith("#");
    }

    // Create lookup maps for both files
    const preGradNodeToLines = {};
    const postGradNodeToLines = {};
    const pyKernelToLines = {};
    const cppCodeToLines = {};

    // Build pre_grad graph lookup map
    preGradGraphData.forEach((line, i) => {
        if (validLine(line)) {
            // Split on '=' and take everything before it
            const beforeEquals = line.trim().split("=")[0];
            // Split on ':' and take everything before it
            const nodeName = beforeEquals.split(":")[0].trim();
            if (nodeName) {
                preGradNodeToLines[nodeName] = i + 1;  // 1-based line numbers
            }
        }
    });

    // Build post_grad lookup map
    postGradGraphData.forEach((line, i) => {
        if (validLine(line)) {
            // Split on '=' and take everything before it
            const beforeEquals = line.trim().split("=")[0];
            // Split on ':' and take everything before it
            const nodeName = beforeEquals.split(":")[0].trim();
            if (nodeName) {
                postGradNodeToLines[nodeName] = i + 1;  // 1-based line numbers
            }
        }
    });

    // Build generated python code lookup map
    let currentKernelName = null;
    let currentKernelLines = [];

    if (codeData) {
        codeData.forEach((line, i) => {
            if (validLine(line)) {
                if (line.includes('async_compile.triton(')) {
                    currentKernelName = line.split('=')[0].trim();
                    currentKernelLines = [i + 1];  // Start collecting lines
                } else if (line.includes("''', device_str='cuda')") && currentKernelName) {
                    currentKernelLines.push(i + 1);  // Add the last line
                    pyKernelToLines[currentKernelName] = currentKernelLines;
                    currentKernelName = null;
                    currentKernelLines = [];
                } else if (currentKernelName) {
                    currentKernelLines.push(i + 1);  // Add lines in between
                }
            }
        });
    }

    if (cppCodeData) {
        let kernelNames = Object.keys(nodeMappings["cppCodeToPost"]);

        // Build generated cpp wrapper code lookup map
        for (let i = 0; i < cppCodeData.length; i++) {
            const line = cppCodeData[i];
            // check if the line include any of the kernel names
            if (validLine(line) && kernelNames.some(kernelName => line.includes(kernelName + "("))) {
                // let kernelName be the first match
                const kernelName = kernelNames.find(kernelName => line.includes(kernelName + "("));
                // create an array for the kernel name if it doesn't exist
                if (!cppCodeToLines[kernelName]) {
                    cppCodeToLines[kernelName] = [];
                }
                // add the line number to the array
                cppCodeToLines[kernelName].push(i + 1);
            }
        }
    }

    // Process all mappings
    const linePreToPost = {};
    const linePostToPre = {};
    const linePyCodeToPost = {};
    const linePostToPyCode = {};
    const lineCppCodeToPost = {};
    const linePostToCppCode = {};

    // Process preToPost using lookup maps
    for (const [fxNodeName, genCodeNodes] of Object.entries(nodeMappings["preToPost"])) {
        if (fxNodeName in preGradNodeToLines) {
            const fxLineNum = preGradNodeToLines[fxNodeName];
            linePreToPost[fxLineNum] = [];
            for (const genNodeName of genCodeNodes) {
                if (genNodeName in postGradNodeToLines) {
                    linePreToPost[fxLineNum].push(postGradNodeToLines[genNodeName]);
                }
            }
        }
    }

    // Process postToPre using lookup maps
    for (const [genNodeName, fxNodeNames] of Object.entries(nodeMappings["postToPre"])) {
        if (genNodeName in postGradNodeToLines) {
            const genLineNum = postGradNodeToLines[genNodeName];
            linePostToPre[genLineNum] = [];
            for (const fxNodeName of fxNodeNames) {
                if (fxNodeName in preGradNodeToLines) {
                    linePostToPre[genLineNum].push(preGradNodeToLines[fxNodeName]);
                }
            }
        }
    }

    // Process pyCodeToPost using lookup maps
    for (const [pyKernelName, postGradNodeNames] of Object.entries(nodeMappings["cppCodeToPost"] || {})) {
        if (pyKernelName in pyKernelToLines) {
            const genLineNums = pyKernelToLines[pyKernelName];
            for (const genLineNum of genLineNums) {
                if (!(genLineNum in linePyCodeToPost)) {
                    linePyCodeToPost[genLineNum] = [];
                }
                for (const postGradNodeName of postGradNodeNames) {
                    if (postGradNodeName in postGradNodeToLines) {
                        linePyCodeToPost[genLineNum].push(postGradNodeToLines[postGradNodeName]);
                    }
                }
            }
        }
    }

    // Process postToPyCode using lookup maps
    for (const [postGradNode, pyKernelNames] of Object.entries(nodeMappings["postToCppCode"] || {})) {
        if (postGradNode in postGradNodeToLines) {
            const genLineNum = postGradNodeToLines[postGradNode];
            linePostToPyCode[genLineNum] = [];
            for (const pyKernelName of pyKernelNames) {
                if (pyKernelName in pyKernelToLines) {
                    linePostToPyCode[genLineNum].push(...pyKernelToLines[pyKernelName]);
                }
            }
        }
    }

    // Process cppCodeToPost using lookup maps
    for (const [cppCodeKernelName, postGradNodeNames] of Object.entries(nodeMappings["cppCodeToPost"])) {
        if (cppCodeKernelName in cppCodeToLines) {
            const genLineNums = cppCodeToLines[cppCodeKernelName];
            for (const genLineNum of genLineNums) {
                if (!(genLineNum in lineCppCodeToPost)) {
                    lineCppCodeToPost[genLineNum] = [];
                }
                for (const postGradNodeName of postGradNodeNames) {
                    if (postGradNodeName in postGradNodeToLines) {
                        lineCppCodeToPost[genLineNum].push(postGradNodeToLines[postGradNodeName]);
                    }
                }
            }
        }
    }

    // Process postToCppCode using lookup maps
    for (const [postGradNode, cppCodeKernelNames] of Object.entries(nodeMappings["postToCppCode"])) {
        if (postGradNode in postGradNodeToLines) {
            const genLineNum = postGradNodeToLines[postGradNode];
            linePostToCppCode[genLineNum] = [];
            for (const cppCodeKernelName of cppCodeKernelNames) {
                if (cppCodeKernelName in cppCodeToLines) {
                    linePostToCppCode[genLineNum].push(...cppCodeToLines[cppCodeKernelName]);
                }
            }
        }
    }

    // Update global variables
    preToPost = linePreToPost;
    postToPre = linePostToPre;
    pyCodeToPost = linePyCodeToPost;
    postToPyCode = linePostToPyCode;
    cppCodeToPost = lineCppCodeToPost;
    postToCppCode = linePostToCppCode;

    console.log('Mappings converted to line numbers:', {
        preToPost,
        postToPre,
        pyCodeToPost,
        postToPyCode,
        cppCodeToPost,
        postToCppCode
    });
}


// Setup editor content
function setupEditorContent(editorId, lines) {
    if (!lines) return;
    
    const editor = document.getElementById(editorId);
    if (!editor) return;

    editor.innerHTML = '';  // Clear existing content
    
    lines.forEach((line, index) => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'line';
        
        // Create text nodes instead of using innerHTML
        const lineNumber = document.createElement('span');
        lineNumber.className = 'line-number';
        lineNumber.textContent = index + 1;
        
        const lineContent = document.createElement('span');
        lineContent.className = 'line-content';
        lineContent.textContent = line;
        
        // Check if this line has any matches
        const lineNum = index + 1;
        let hasMatch = false;
        switch (editorId) {
            case 'preGradGraph':
                hasMatch = preToPost[lineNum] && preToPost[lineNum].length > 0;
                break;
            case 'postGradGraph':
                hasMatch = (postToPre[lineNum] && postToPre[lineNum].length > 0) ||
                          (postToPyCode[lineNum] && postToPyCode[lineNum].length > 0) ||
                          (postToCppCode[lineNum] && postToCppCode[lineNum].length > 0);
                break;
            case 'generatedCode':
                hasMatch = (pyCodeToPost[lineNum] && pyCodeToPost[lineNum].length > 0) || 
                (cppCodeToPost[lineNum] && cppCodeToPost[lineNum].length > 0);
                break;
        }
        
        if (hasMatch) {
            lineContent.classList.add('has-match');
        }
        
        lineDiv.appendChild(lineNumber);
        lineDiv.appendChild(lineContent);
        
        // Add both click and hover handlers
        lineDiv.addEventListener('click', () => handleLineClick(editorId, index + 1));
        lineDiv.addEventListener('mouseenter', () => handleLineHover(editorId, index + 1));
        lineDiv.addEventListener('mouseleave', clearHighlights);
        
        editor.appendChild(lineDiv);
    });
}

// Handle line hover
function handleLineHover(editorId, lineNumber) {
    // Clear previous highlights
    clearHighlights();
    
    // Add highlight to hovered line
    const hoveredLine = document.querySelector(`#${editorId} .line:nth-child(${lineNumber})`);
    if (hoveredLine) {
        hoveredLine.classList.add('highlight');
        // Remove scrolling for hovered panel
    }

    // Highlight and scroll corresponding lines
    highlightCorrespondingLines(editorId, lineNumber);
}

// Clear all highlights
function clearHighlights() {
    document.querySelectorAll('.line').forEach(line => {
        line.classList.remove('highlight');
    });
}

// Update handleLineClick to use the same pattern
function handleLineClick(editorId, lineNumber) {
    clearHighlights();
    
    // Add highlight to clicked line
    const clickedLine = document.querySelector(`#${editorId} .line:nth-child(${lineNumber})`);
    if (clickedLine) {
        clickedLine.classList.add('highlight');
        clickedLine.scrollIntoView({
            behavior: 'smooth',
            block: 'center',
            inline: 'nearest'
        });
    }

    // Highlight corresponding lines
    highlightCorrespondingLines(editorId, lineNumber);
}

// Initialize data from pre-embedded content
function initializeData() {
    try {
        // Get content from pre tags
        const preGradGraph = document.querySelector('#preGradGraph pre');
        const postGradGraph = document.querySelector('#postGradGraph pre');
        const generatedCode = document.querySelector('#generatedCode pre');

        if (preGradGraph) preGradGraphData = preGradGraph.textContent.split('\n');
        if (postGradGraph) postGradGraphData = postGradGraph.textContent.split('\n');
        if (generatedCode) {
            const content = generatedCode.textContent;
            if (content.includes('async_compile.triton(')) {
                // This is Python code
                codeData = content.split('\n');
                cppCodeData = null;
            } else {
                // This is C++ code
                cppCodeData = content.split('\n');
                codeData = null;
            }
        }

        // Convert node mappings to line numbers
        convertNodeMappingsToLineNumbers();

        // Setup highlighting
        setupEditorContent('preGradGraph', preGradGraphData);
        setupEditorContent('postGradGraph', postGradGraphData);
        setupEditorContent('generatedCode', codeData || cppCodeData);

        // If it's C++ code, scroll to run_impl
        if (cppCodeData) {
            const cppEditor = document.getElementById('generatedCode');
            if (cppEditor) {
                const targetLine = Array.from(cppEditor.querySelectorAll('.line')).find(
                    line => line.textContent.includes('void AOTInductorModel::run_impl(')
                );
                if (targetLine) {
                    targetLine.scrollIntoView({ behavior: 'auto', block: 'center' });
                }
            }
        }
    } catch (error) {
        console.error('Error initializing data:', error);
        console.error(error.stack);
    }
}

// Call initialization when the page loads
window.addEventListener('DOMContentLoaded', initializeData);

// Highlight corresponding lines
function highlightCorrespondingLines(sourceEditorId, lineNumber) {
    let correspondingLines = findCorrespondingLines(sourceEditorId, lineNumber);
    
    Object.entries(correspondingLines).forEach(([editorId, lines]) => {
        // Skip scrolling if this is the source editor
        if (lines && editorId !== sourceEditorId) {
            // Handle both single numbers and arrays of numbers
            const lineNumbers = Array.isArray(lines) ? lines : [lines];
            
            // Get the middle line number for scrolling
            const middleIndex = Math.floor(lineNumbers.length / 2);
            let hasScrolled = false;
            
            lineNumbers.forEach((line, index) => {
                const lineElement = document.querySelector(`#${editorId} .line:nth-child(${line})`);
                if (lineElement) {
                    lineElement.classList.add('highlight');
                    
                    // Scroll to the middle line of the highlighted range
                    if (index === middleIndex && !hasScrolled) {
                        lineElement.scrollIntoView({
                            behavior: 'smooth',
                            block: 'center',
                            inline: 'nearest'
                        });
                        hasScrolled = true;
                    }
                }
            });
        }
    });
}

// Given a line in sourceEditorId, find the corresponding lines in the other editors that should be highlighted.
function findCorrespondingLines(sourceEditorId, lineNumber) {
    let result = {};
    
    switch (sourceEditorId) {
        case 'preGradGraph':
            result.postGradGraph = preToPost[lineNumber] || [];
            if (result.postGradGraph.length > 0) {
                result.generatedCode = [];
                for (const postLine of result.postGradGraph) {
                    if (codeData) {
                        if (postToPyCode[postLine]) {
                            result.generatedCode.push(...postToPyCode[postLine]);
                        }
                    } else {
                        if (postToCppCode[postLine]) {
                            result.generatedCode.push(...postToCppCode[postLine]);
                        }
                    }
                }
            }
            break;
            
        case 'postGradGraph':
            result.preGradGraph = postToPre[lineNumber] || [];
            if (codeData) {
                result.generatedCode = postToPyCode[lineNumber] || [];
            } else {
                result.generatedCode =  postToCppCode[lineNumber] || [];
            }
            break;
            
        case 'generatedCode':
            if (codeData) {
                // Python code
                result.postGradGraph = pyCodeToPost[lineNumber] || [];
            } else {
                result.postGradGraph = cppCodeToPost[lineNumber] || [];
            }
            if (result.postGradGraph.length > 0) {
                result.preGradGraph = [];
                for (const postLine of result.postGradGraph) {
                    if (postToPre[postLine]) {
                        result.preGradGraph.push(...postToPre[postLine]);
                    }
                }
            }
            break;
    }
    
    return result;
}