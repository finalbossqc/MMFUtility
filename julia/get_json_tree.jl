using JSON

"""
    print_json_tree(data, counter; indent=0, max_lines=20)

Recursively prints a JSON object or array as a tree structure,
but stops after `max_lines` lines have been printed.
"""
function print_json_tree(data, counter::Ref{Int}; indent=0, max_lines=20)
    if counter[] >= max_lines
        return
    end

    prefix = "  " ^ indent

    if isa(data, Dict)
        for (key, value) in data
            if counter[] >= max_lines
                return
            end
            println(prefix * "📁 $key")
            counter[] += 1
            print_json_tree(value, counter; indent = indent + 1, max_lines = max_lines)
        end
    elseif isa(data, Vector)
        for (i, item) in enumerate(data)
            if counter[] >= max_lines
                return
            end
            println(prefix * "🔢 [$i]")
            counter[] += 1
            print_json_tree(item, counter; indent = indent + 1, max_lines = max_lines)
        end
    else
        println(prefix * "📄 $data")
        counter[] += 1
    end
end

# Read and parse JSON file
function parse_and_print_json(filename::String; max_lines=20)
    json_string = read(filename, String)
    json_data = JSON.parse(json_string)
    counter = Ref(0)
    print_json_tree(json_data, counter; max_lines=max_lines)
end