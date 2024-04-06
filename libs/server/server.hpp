#pragma once

#include <boost/asio.hpp>
#include <vector>
#include <mutex>
#include <cstdint>

using boost::asio::ip::tcp;

/**
*
* @brief The server which receives data from the client
*
* An object which represents the server that handles all the data
* from the client(s).
*
* @author Tristan Huen
*
*/
class Server {
public:

    std::vector<int> data = std::vector<int>(4);///< Storage for data.
    std::mutex& mutex;

    /** @brief Constructor for server object
    *
    * @param io_context The IO context
    * @param port The port number to use
    * @return None
    */
    Server(boost::asio::io_context& io_context, unsigned short port, std::mutex& mutex);

    void run();
    void stop();

private:

    tcp::acceptor _acceptor; ///< Acceptor object
    boost::asio::io_context& _io_context; ///< IO context object

    /** @brief Accepts incoming connection from client
    *
    * @return None
    */
    void do_accept();
};