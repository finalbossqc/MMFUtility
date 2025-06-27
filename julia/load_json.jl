using JSON 

function load_json(file::String)
    data = Nothing
    try
        file_content = read(file, String)
        data = JSON.parse(file_content)
        println("JSON data loaded from file successfully!")
    catch e
        if isa(e, SystemError) && occursin("no such file or directory", e.msg)
            println("Error: data.json not found.")
        elseif isa(e, JSON.Parser.JSONParseException)
            println("Error: Could not decode JSON from data.json. Check file format.")
        else
            println("An unexpected error occurred: ", e)
        end
    end

    return data
end