#pragma once

#include <boost/asio.hpp>

using boost::asio::ip::tcp;

/*
NOTE: This code is not used at the moment. It was supposed to
      be used for video streaming from the Pi but that is now
      using rpicam-vid and gstreamer instead. This code remains here
      in case any other kind of data needs to be retrieved from
      the Pi      
*/

/**
*
* @brief Represents a user for connection to a server
*
* An object which represents a user/client that connects to the server. User can
* read/write unsigned vectors of size 4096.
*
* @author Tristan Huen
*
*/
class Client {
public:

    /** @brief Writes data to the server.
    *
    * @param io_context The IO context object
    * @param ip The IP address to connect to
    * @param port The port number
    * @return None
    */
    Client(boost::asio::io_context& io_context, std::string ip, unsigned short port);

    /** @brief Connects the client to the server.
    *
    * @return None
    */
    void connect();

    /** @brief Writes data to the server.
    *
    * @param request The request data from the server.
    * @return None
    * 
    * NOTE: This code will be changed.
    * 
    */
    void write(unsigned request);

    /** @brief Reads an image frame back from the server
    *
    * @param values The values of the image frame.
    * @return None
    */
    void read(std::vector<std::uint8_t>& values);

private:
    boost::asio::io_context& _io_context;  ///< The IO context object.
    tcp::endpoint _endpoint;               ///< The endpoint object.
    std::unique_ptr<tcp::socket> _socket;  ///< A unique pointer to the socket. 
};
